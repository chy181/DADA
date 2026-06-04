import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    """权重非负的线性层。

    SelfImp 中的统一异常强度 `z` 需要通过一个简单映射解释多路原始评分。
    这里用 `softplus(weight_raw)` 约束权重为非负，使映射保持单调方向：
    `z` 越大，各评分通道的预测值不会因为负权重而反向降低。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.weight_raw)
        return F.linear(x, weight, self.bias)


class MonotonicCalibrator(nn.Module):
    """从统一异常强度到多路评分的单调校准器。

    对应技术方案中“统一评分需要经过简单映射后同时还原各路原始评分”的
    接口。输入为 `[B, T, 1]` 的点级隐变量 `z`，输出为 `[B, T, S]` 的多路
    分数重构，其中 `S` 是重构、频域、WGAD 等评分通道数。
    """

    def __init__(self, n_scores: int, hidden_dim: int):
        super().__init__()
        self.fc1 = PositiveLinear(1, hidden_dim)
        self.fc2 = PositiveLinear(hidden_dim, n_scores)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.softplus(self.fc1(z))
        return self.fc2(h)


class RawToLatentEncoder(nn.Module):
    """从标准化原始窗口生成统一异常强度先验。

    该分支为测试时训练提供可选 raw prior，使优化出的 `z` 不只依赖已有
    异常评分，也能参考当前窗口的原始多变量形态。其权重由
    `self_imp_raw_weight` 控制；设置为 0 时该先验不影响目标函数。
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        return F.softplus(self.fc2(h))


class Plugin(nn.Module):
    """基于测试时训练的自解释多异常评分聚合插件。

    对应《二阶段技术方案》中的“基于测试时训练的自解释多评分聚合”。该模块
    不在训练阶段学习固定融合权重，而是在每个测试窗口内临时优化一个点级
    统一异常强度 `z`。优化目标要求 `z` 经过单调校准器后能够同时重构多路
    原始异常评分，从而提取重构误差、频域误差、WGAD 关联异常分数等评分中
    共同响应的异常信号。

    设计约束：
        - `score_channels` 至少包含两个评分通道，形状为 `[B, T, S]`。
        - 输出 `score` 是 `[B, T]` 的点级聚合分数，可继续与主分数融合。
        - `self_imp_l1` 控制稀疏性，`self_imp_tv` 控制时间连续性，
          `self_imp_raw_weight` 控制原始序列先验约束。
    """

    def __init__(self, args):
        super().__init__()
        self.steps = args.self_imp_steps
        self.lr = args.self_imp_lr
        self.l1 = args.self_imp_l1
        self.tv = args.self_imp_tv
        self.huber_delta = args.self_imp_huber_delta
        self.hidden_dim = args.self_imp_hidden_dim
        self.raw_hidden_dim = args.self_imp_raw_hidden_dim
        self.raw_weight = args.self_imp_raw_weight
        self.seed = args.self_imp_seed
        self.eps = 1e-6

    def inference(self, raw_window: torch.Tensor, score_channels: torch.Tensor):
        """在当前测试窗口上优化自解释聚合分数。

        Args:
            raw_window: 原始局部窗口，形状为 `[B, T, C]`。仅用于生成可选
                raw prior，进入本模块前会 detach，不反传到主干模型。
            score_channels: 多路异常评分，形状为 `[B, T, S]`。通常包含重构
                分数、频域分数，以及启用 WGAD 时的关联异常分数。

        Returns:
            `{"score": score}`，其中 `score` 为 `[B, T]`。该分数不是固定加权
            和，而是测试时训练得到的统一异常表示。
        """
        if score_channels.ndim != 3 or score_channels.shape[-1] < 2:
            raise ValueError("self-imp requires score_channels with shape [B, T, S] and S >= 2")

        raw_window = raw_window.detach()
        score_channels = score_channels.detach()

        with torch.enable_grad():
            torch.manual_seed(int(self.seed))
            if raw_window.is_cuda:
                torch.cuda.manual_seed_all(int(self.seed))

            # 窗口级标准化阶段：
            # 只在当前测试窗口内部统计均值和方差，降低不同窗口幅值差异对
            # raw prior 学习的影响；已有评分通道保持原尺度，避免改变检测分支
            # 已经形成的分数语义。
            raw_mean = raw_window.mean(dim=1, keepdim=True)
            raw_std = raw_window.std(dim=1, keepdim=True, unbiased=False)
            raw_tensor = (raw_window - raw_mean) / (raw_std + self.eps)
            score_tensor = score_channels

            # 统一异常强度初始化阶段：
            # 技术方案中把多路评分视为真实异常强度的含噪观测，因此初值采用
            # 多路评分均值，让优化从共同响应最强的位置开始。后续只在当前
            # 窗口内校准该点级异常强度，不更新主干模型或其它检测分支。
            init_z = score_tensor.mean(dim=-1, keepdim=True).clamp_min(0.0)
            z_raw = nn.Parameter(init_z.clone())
            calibrator = MonotonicCalibrator(
                n_scores=score_tensor.shape[-1],
                hidden_dim=int(self.hidden_dim),
            ).to(score_tensor.device)
            raw_encoder = RawToLatentEncoder(
                input_dim=raw_tensor.shape[-1],
                hidden_dim=int(self.raw_hidden_dim),
            ).to(score_tensor.device)

            optimizer = torch.optim.Adam(
                [z_raw, *calibrator.parameters(), *raw_encoder.parameters()],
                lr=float(self.lr),
            )

            for _ in range(int(self.steps)):
                optimizer.zero_grad()
                z = F.softplus(z_raw)
                pred = calibrator(z)
                raw_prior = raw_encoder(raw_tensor)

                # 多评分自解释重构阶段：
                # 单一 `z` 必须通过单调校准器同时解释所有评分通道。如果某一路
                # 分数是噪声而其它分数没有共同响应，Huber 重构会降低其对最终
                # 统一异常强度的主导作用。
                recon_loss = F.huber_loss(
                    pred,
                    score_tensor,
                    delta=float(self.huber_delta),
                    reduction="mean",
                )
                sparse_loss = z.mean()
                if z.shape[1] > 1:
                    tv_loss = torch.abs(z[:, 1:] - z[:, :-1]).mean()
                else:
                    tv_loss = torch.tensor(0.0, device=z.device)
                raw_prior_loss = F.mse_loss(z, raw_prior, reduction="mean")

                # 测试时训练目标汇总阶段：
                # recon_loss 保证可解释多评分，sparse_loss 偏向稀疏异常，
                # tv_loss 偏向时间连续异常，raw_prior_loss 用原始序列形态约束
                # `z`。这些优化仅发生在当前窗口内。
                loss = (
                    recon_loss
                    + float(self.l1) * sparse_loss
                    + float(self.tv) * tv_loss
                    + float(self.raw_weight) * raw_prior_loss
                )
                loss.backward()
                optimizer.step()

            # 聚合分数输出阶段：
            # 只返回优化后的非负 `z`，外部再按 `self_imp_alpha` 与当前主分数融合。
            score = F.softplus(z_raw).detach().squeeze(-1)
        return {"score": score}

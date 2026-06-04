import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class PatchHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(x))


class DynamicGraphGCN(nn.Module):
    """单 patch 内的动态通道图。

    这是 WGAD 的兼容路径：当 `wgad_window_size == 1` 时使用，行为保持为
    当前仓库原有实现。输入节点只包含同一 patch 内的变量/通道，因此邻接矩阵
    形状为 `[B, P, C, C]`，表示每个 patch 上独立学习到的通道关系。
    """

    def __init__(self, d_model, num_layers=1, dropout=0.1):
        super().__init__()
        self.graph_query = nn.Linear(d_model, d_model // 2)
        self.graph_key = nn.Linear(d_model, d_model // 2)
        self.gcn_weights = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.graph_query(x)
        k = self.graph_key(x)
        adj_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        adj_matrix = torch.softmax(adj_score, dim=-1)

        h = x
        for gcn_weight, norm in zip(self.gcn_weights, self.norms):
            h_agg = torch.matmul(adj_matrix, h)
            h_trans = F.gelu(gcn_weight(h_agg))
            h_trans = self.dropout(h_trans)
            h = norm(h + h_trans)
        return h, adj_matrix


class SpatioTemporalGraphGCN(nn.Module):
    """二级 patch 窗口内的自适应时空图。

    对应技术方案“双域时空关联一致性检测”中的片段级时空邻接图构建。
    输入表征形状为 `[B, P, C, D]`，其中 `P` 是 patch 数、`C` 是变量数。
    当 `window_size > 1` 时，模块在 patch 维度上滑动二级窗口，并把窗口内
    `window_size * C` 个“时间片段-变量”组合展开为图节点，得到形状为
    `[B, window_count, window_size*C, window_size*C]` 的邻接矩阵。

    该图同时包含变量之间和相邻片段之间的自适应关联。GCN 更新后的窗口节点
    会被平均回原 patch 网格，保证后续重构头和预测头仍复用 `[B, P, C, D]`
    接口。
    """

    def __init__(self, d_model, num_vars, window_size, window_stride=1, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.window_stride = window_stride
        self.graph_query = nn.Linear(d_model, d_model // 2)
        self.graph_key = nn.Linear(d_model, d_model // 2)
        self.time_embedding = nn.Parameter(torch.zeros(window_size, d_model))
        self.var_embedding = nn.Parameter(torch.zeros(num_vars, d_model))
        self.gcn_weights = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.var_embedding, std=0.02)

    def forward(self, x):
        bsz, patch_num, n_vars, d_model = x.shape
        if n_vars != self.num_vars:
            raise ValueError(f"WGAD expected {self.num_vars} variables, got {n_vars}")
        if patch_num < self.window_size:
            raise ValueError(
                f"WGAD graph window_size={self.window_size} exceeds patch_num={patch_num}"
            )

        # 二级时空窗口构建：
        # 在 patch 维度上滑动窗口，把相邻片段与变量组合成局部时空子图。
        # 这一步把 `[B, P, C, D]` 转换为 `[B, window_count, W, C, D]`。
        windows = x.unfold(dimension=1, size=self.window_size, step=self.window_stride)
        windows = windows.permute(0, 1, 4, 2, 3).contiguous()
        window_count = windows.shape[1]

        # 节点位置注入：
        # 时间位置 embedding 区分同一变量在二级窗口内的不同片段，变量
        # embedding 区分同一片段内的不同通道，避免展开节点后丢失时空身份。
        windows = windows + self.time_embedding.view(1, 1, self.window_size, 1, d_model)
        windows = windows + self.var_embedding.view(1, 1, 1, n_vars, d_model)
        nodes = windows.view(bsz, window_count, self.window_size * n_vars, d_model)

        # 自适应邻接矩阵生成：
        # 对每个二级窗口内的 W*C 个节点计算 query-key 相关性，得到局部时空图。
        # 该邻接既可表达变量间关系，也可表达跨片段的动态依赖。
        q = self.graph_query(nodes)
        k = self.graph_key(nodes)
        adj_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        adj_matrix = torch.softmax(adj_score, dim=-1)

        # 图消息传递：
        # 用学习到的邻接对节点表征做聚合和残差更新，输出仍保留每个时空节点
        # 的特征，后续再回填到 patch 网格供重构/预测头使用。
        h = nodes
        for gcn_weight, norm in zip(self.gcn_weights, self.norms):
            h_agg = torch.matmul(adj_matrix, h)
            h_trans = F.gelu(gcn_weight(h_agg))
            h_trans = self.dropout(h_trans)
            h = norm(h + h_trans)
        patch_grid = self._windows_to_patch_grid(h, bsz, patch_num, n_vars)
        return patch_grid, adj_matrix

    def _windows_to_patch_grid(self, window_nodes, bsz, patch_num, n_vars):
        window_count = window_nodes.shape[1]
        d_model = window_nodes.shape[-1]
        windows = window_nodes.view(bsz, window_count, self.window_size, n_vars, d_model)
        out = window_nodes.new_zeros(bsz, patch_num, n_vars, d_model)
        counts = window_nodes.new_zeros(1, patch_num, 1, 1)

        # 重叠窗口回填：
        # 同一个 patch 可能被多个二级窗口覆盖，因此把所有覆盖该 patch 的
        # 更新表征累加后取均值，恢复为 `[B, P, C, D]`。
        for win_idx in range(window_count):
            start = win_idx * self.window_stride
            end = start + self.window_size
            out[:, start:end] += windows[:, win_idx]
            counts[:, start:end] += 1
        return out / counts.clamp_min(1.0)


class Db2SWTExtractor(nn.Module):
    def __init__(self, level=3):
        super().__init__()
        self.level = level
        s3 = math.sqrt(3)
        s2 = math.sqrt(2)
        h0 = (1 + s3) / (4 * s2)
        h1 = (3 + s3) / (4 * s2)
        h2 = (3 - s3) / (4 * s2)
        h3 = (1 - s3) / (4 * s2)
        g0 = h3
        g1 = -h2
        g2 = h1
        g3 = -h0
        self.register_buffer(
            "low_pass", torch.tensor([[[h0, h1, h2, h3]]], dtype=torch.float32)
        )
        self.register_buffer(
            "high_pass", torch.tensor([[[g0, g1, g2, g3]]], dtype=torch.float32)
        )

    def forward(self, x):
        approx = x
        features = []
        with torch.no_grad():
            for i in range(self.level):
                dilation = 2 ** i
                pad_size = 3 * dilation
                padded = F.pad(approx, (pad_size, 0), mode="replicate")
                detail = F.conv1d(padded, self.high_pass, dilation=dilation)
                approx = F.conv1d(padded, self.low_pass, dilation=dilation)
                features.append(detail)
            features.append(approx)
            return torch.cat(features, dim=1)


class WaveletPatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, level, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_channels = level + 1
        self.swt_extractor = Db2SWTExtractor(level=level)
        self.value_embedding = nn.Linear(self.n_channels * patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs, seq_len, n_vars = x.shape
        x = x.permute(0, 2, 1).contiguous().view(bs * n_vars, 1, seq_len)
        swt_features = self.swt_extractor(x)
        patches = swt_features.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patch_num = patches.shape[1]
        patches_flat = patches.view(
            bs * n_vars, patch_num, self.n_channels * self.patch_len
        )
        out = self.value_embedding(patches_flat)
        return self.dropout(out), n_vars, patches_flat


class TimeDomainExecutor(nn.Module):
    """时间域 WGAD 执行器。

    该分支直接使用主干模型的 patch embedding 表征原始时间序列历史段 `z`，
    再通过通道图或时空图建模变量关联。它承担两个辅助任务：
    1. 重构历史上下文，用 `time_loss_rec` 捕捉当前窗口内的异常偏离；
    2. 预测未来片段，用 `time_loss_pre` 捕捉动态演化关系异常。

    返回的邻接矩阵会与时频域分支邻接矩阵做图一致性对比学习。
    """

    def __init__(
        self,
        backbone,
        d_model,
        num_vars,
        patch_len,
        stride,
        context_size,
        horizon,
        head_dropout,
        dropout,
        gcn_layers,
        graph_window_size,
        graph_window_stride,
    ):
        super().__init__()
        object.__setattr__(self, "_backbone", backbone)
        self.patch_len = patch_len
        self.patch_num = int((context_size - patch_len) / stride + 1)
        self.horizon = horizon
        self.graph_window_size = graph_window_size
        if self.graph_window_size == 1:
            self.dynamic_gcn = DynamicGraphGCN(
                d_model=d_model, num_layers=gcn_layers, dropout=dropout
            )
        else:
            self.st_gcn = SpatioTemporalGraphGCN(
                d_model=d_model,
                num_vars=num_vars,
                window_size=graph_window_size,
                window_stride=graph_window_stride,
                num_layers=gcn_layers,
                dropout=dropout,
            )
        self.time_rec_head = PatchHead(
            nf=d_model, target_window=patch_len, head_dropout=head_dropout
        )
        self.time_pre_head = FlattenHead(
            nf=self.patch_num * d_model,
            target_window=horizon,
            head_dropout=head_dropout,
        )

    def forward(self, z, x):
        """计算时间域历史重构误差、未来预测误差和图邻接矩阵。

        Args:
            z: 历史上下文，形状为 `[B, context_size, C]`。
            x: 未来预测目标，形状为 `[B, horizon, C]`。

        Returns:
            `(loss_rec, loss_pre, adj_matrix)`，其中前两个 loss 保留点级和通道
            维度，供训练求均值、推理转换为异常评分。
        """
        bs, _, n_vars = z.shape
        time_emb, _ = self._backbone.patch_embed_channels(z)
        time_emb = (
            time_emb.view(bs, n_vars, self.patch_num, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # 时间域图建模阶段：
        # `wgad_window_size=1` 时建 patch 内通道图；大于 1 时建局部时空图。
        # 两条路径都输出 patch 网格表征，保证下游重构和预测接口一致。
        if self.graph_window_size == 1:
            gcn_out, adj_matrix = self.dynamic_gcn(time_emb)
        else:
            gcn_out, adj_matrix = self.st_gcn(time_emb)

        # 历史重构阶段：
        # 用图更新后的历史表征重构上下文 `z`，该误差用于捕捉当前窗口内
        # 不符合正常关联模式的局部异常。
        rec_out = self.time_rec_head(gcn_out)
        rec_out = rearrange(
            rec_out, "bs patch_num n_vars patch_len -> bs (patch_num patch_len) n_vars"
        )
        loss_rec = F.mse_loss(rec_out, z, reduction="none")

        # 未来预测阶段：
        # 将同一变量的所有 patch 表征展平后预测未来 `horizon`，该误差用于
        # 捕捉时序演化关系异常。
        pre_input = gcn_out.permute(0, 2, 1, 3).contiguous()
        pre_out = self.time_pre_head(pre_input).permute(0, 2, 1).contiguous()
        loss_pre = F.mse_loss(pre_out, x, reduction="none")
        return loss_rec, loss_pre, adj_matrix


class WaveletDomainExecutor(nn.Module):
    """时频域 WGAD 执行器。

    该分支先用 db2 SWT 将每个变量的历史上下文分解为多尺度时频表示，再在
    patch 级时频表征上构建通道图或时空图。它与时间域分支保持相同的重构
    与预测任务，但误差发生在小波分解后的多尺度空间中，用于补充幅值变化
    不明显、频域结构变化更敏感的异常证据。
    """

    def __init__(
        self,
        d_model,
        num_vars,
        patch_len,
        stride,
        context_size,
        horizon,
        head_dropout,
        dropout,
        wave_scales,
        gcn_layers,
        graph_window_size,
        graph_window_stride,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_num = int((context_size - patch_len) / stride + 1)
        self.n_wave_channels = wave_scales + 1
        self.horizon = horizon
        self.wavelet_embedding = WaveletPatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            level=wave_scales,
            dropout=dropout,
        )
        self.graph_window_size = graph_window_size
        if self.graph_window_size == 1:
            self.dynamic_gcn = DynamicGraphGCN(
                d_model=d_model, num_layers=gcn_layers, dropout=dropout
            )
        else:
            self.st_gcn = SpatioTemporalGraphGCN(
                d_model=d_model,
                num_vars=num_vars,
                window_size=graph_window_size,
                window_stride=graph_window_stride,
                num_layers=gcn_layers,
                dropout=dropout,
            )
        self.wavelet_rec_head = PatchHead(
            nf=d_model,
            target_window=self.n_wave_channels * patch_len,
            head_dropout=head_dropout,
        )
        self.wavelet_pre_head = FlattenHead(
            nf=self.patch_num * d_model,
            target_window=self.n_wave_channels * horizon,
            head_dropout=head_dropout,
        )

    def forward(self, z, x):
        """计算时频域历史重构误差、未来预测误差和图邻接矩阵。

        Args:
            z: 历史上下文，形状为 `[B, context_size, C]`。
            x: 未来预测目标，形状为 `[B, horizon, C]`。

        Returns:
            `(loss_rec, loss_pre, adj_matrix)`。时频域重构和预测 loss 额外包含
            小波尺度维度，后续会在推理阶段对通道和尺度求均值形成点级评分。
        """
        bs, _, n_vars = z.shape
        emb_flat, _, gt_wavelet_patches = self.wavelet_embedding(z)
        wavelet_emb = (
            emb_flat.view(bs, n_vars, self.patch_num, -1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # 时频域图建模阶段：
        # 先在小波多尺度 patch 表征上建图，再输出与时间域同形状的 patch
        # 网格表征。该分支用于补充频域结构变化带来的异常证据。
        if self.graph_window_size == 1:
            gcn_out, adj_matrix = self.dynamic_gcn(wavelet_emb)
        else:
            gcn_out, adj_matrix = self.st_gcn(wavelet_emb)

        # 时频域历史重构阶段：
        # 重构每个 patch 的多尺度小波系数，衡量历史上下文在时频结构上的偏离。
        rec_out = self.wavelet_rec_head(gcn_out)
        rec_out = rec_out.view(
            bs, self.patch_num, n_vars, self.n_wave_channels, self.patch_len
        )
        rec_out = rearrange(
            rec_out,
            "bs patch_num n_vars n_wave_channels patch_len -> bs (patch_num patch_len) n_vars n_wave_channels",
        )
        gt_rec = gt_wavelet_patches.view(
            bs, n_vars, self.patch_num, self.n_wave_channels, self.patch_len
        )
        gt_rec = gt_rec.permute(0, 2, 1, 3, 4).contiguous()
        gt_rec = rearrange(
            gt_rec,
            "bs patch_num n_vars n_wave_channels patch_len -> bs (patch_num patch_len) n_vars n_wave_channels",
        )
        loss_rec = F.mse_loss(rec_out, gt_rec, reduction="none")

        # 时频域未来预测阶段：
        # 预测未来片段的小波系数，并与真实未来片段的小波分解结果比较。
        # 该误差反映频域演化模式是否符合正常样本规律。
        pre_input = gcn_out.permute(0, 2, 1, 3).contiguous()
        pre_out = self.wavelet_pre_head(pre_input)
        pre_out = pre_out.view(bs, n_vars, self.n_wave_channels, self.horizon)
        pre_out = pre_out.permute(0, 3, 1, 2).contiguous()

        x_flat = x.permute(0, 2, 1).contiguous().view(bs * n_vars, 1, self.horizon)
        with torch.no_grad():
            x_wavelet = self.wavelet_embedding.swt_extractor(x_flat)
        gt_pre = x_wavelet.view(bs, n_vars, self.n_wave_channels, self.horizon)
        gt_pre = gt_pre.permute(0, 3, 1, 2).contiguous()
        loss_pre = F.mse_loss(pre_out, gt_pre, reduction="none")
        return loss_rec, loss_pre, adj_matrix


class Plugin(nn.Module):
    """WGAD 双域时空关联一致性检测插件。

    该插件对应技术方案中的“双域时空关联一致性检测”。输入窗口会被划分为
    历史上下文 `z` 和未来目标 `y`：时间域分支和时频域分支分别构建图、执行
    历史重构与未来预测；两个分支的邻接矩阵通过对比学习对齐，从而学习正常
    样本下稳定的一致关联模式。推理阶段把重构误差、预测误差和双域图不一致
    分数组合为 WGAD 异常评分。

    兼容性约定：
        `wgad_window_size == 1` 时使用原有 patch 内通道图，邻接矩阵为
        `[B, P, C, C]`；`wgad_window_size > 1` 时启用时空图，邻接矩阵为
        `[B, window_count, W*C, W*C]`。因此二级窗口跨度为 1 时可退回旧版本
        的接口和行为。
    """

    def __init__(self, backbone, args, num_features):
        super().__init__()
        self.win_size = getattr(args, "win_size", backbone.win_size)
        self.patch_len = getattr(args, "patch_len", backbone.patch_len)
        self.horizon = args.wgad_horizon
        self.context_size = self.win_size - self.horizon
        if self.context_size <= 0:
            raise ValueError("wgad_horizon must be smaller than win_size")
        if self.context_size % self.patch_len != 0:
            raise ValueError("win_size - wgad_horizon must be divisible by patch_len")

        self.patch_num = self.context_size // self.patch_len
        self.graph_window_size = getattr(args, "wgad_window_size", 1)
        self.graph_window_stride = getattr(args, "wgad_window_stride", 1)
        if self.graph_window_size < 1:
            raise ValueError("wgad_window_size must be at least 1")
        if self.graph_window_size > self.patch_num:
            raise ValueError(
                f"wgad_window_size must be <= patch_num ({self.patch_num}), got {self.graph_window_size}"
            )
        if self.graph_window_stride < 1:
            raise ValueError("wgad_window_stride must be at least 1")
        if self.graph_window_stride != 1:
            raise ValueError("WGAD spatio-temporal graph currently supports wgad_window_stride=1 only")

        self.score_lambda = args.wgad_score_lambda
        self.lambda_cl = args.wgad_lambda_cl
        self.lambda_wavelet = args.wgad_lambda_wavelet
        self.score_mode = args.wgad_score_mode
        self.revin = RevIN(
            num_features,
            affine=args.wgad_affine,
            subtract_last=args.wgad_subtract_last,
        )
        self.time_executor = TimeDomainExecutor(
            backbone=backbone,
            d_model=backbone.hidden_dim,
            num_vars=num_features,
            patch_len=self.patch_len,
            stride=self.patch_len,
            context_size=self.context_size,
            horizon=self.horizon,
            head_dropout=args.wgad_head_dropout,
            dropout=args.wgad_dropout,
            gcn_layers=args.wgad_gcn_layers,
            graph_window_size=self.graph_window_size,
            graph_window_stride=self.graph_window_stride,
        )
        self.wavelet_executor = WaveletDomainExecutor(
            d_model=backbone.hidden_dim,
            num_vars=num_features,
            patch_len=self.patch_len,
            stride=self.patch_len,
            context_size=self.context_size,
            horizon=self.horizon,
            head_dropout=args.wgad_head_dropout,
            dropout=args.wgad_dropout,
            wave_scales=args.wgad_wave_scales,
            gcn_layers=args.wgad_gcn_layers,
            graph_window_size=self.graph_window_size,
            graph_window_stride=self.graph_window_stride,
        )

    def forward(self, x):
        """训练阶段 WGAD loss 入口。

        Args:
            x: 一个局部窗口，形状为 `[B, win_size, C]`。

        Returns:
            `{"loss": loss}`。loss 由时间域重构、时间域预测、加权时频域
            重构/预测，以及双域图一致性对比损失组成。
        """
        x = self.revin(x, "norm")
        z = x[:, : self.context_size, :]
        y = x[:, self.context_size :, :]

        # 双域辅助任务阶段：
        # 时间域和时频域分别产生历史重构误差、未来预测误差和图邻接矩阵。
        # 这些输出同时服务于训练 loss 和推理时的多路异常评分。
        time_loss_rec, time_loss_pre, time_adj = self.time_executor(z, y)
        wavelet_loss_rec, wavelet_loss_pre, wavelet_adj = self.wavelet_executor(z, y)

        # 双域图一致性约束阶段：
        # 同一窗口的时间域图与时频域图作为正样本对，其它窗口作为负样本，
        # 使正常样本下的两种视图学习到一致的关联结构。
        loss_cl = self.compute_graph_contrastive_loss(time_adj, wavelet_adj)
        loss = (
            time_loss_rec.mean()
            + time_loss_pre.mean()
            + self.lambda_wavelet * (wavelet_loss_rec.mean() + wavelet_loss_pre.mean())
            + self.lambda_cl * loss_cl
        )
        return {"loss": loss}

    def inference(self, x):
        """推理阶段 WGAD 分数入口。

        Args:
            x: 一个局部窗口，形状为 `[B, win_size, C]`。

        Returns:
            字典形式的点级分数，核心键为 `score`。辅助键 `time_rec`、
            `time_pre`、`wavelet_rec`、`wavelet_pre` 和 `cl` 用于分析不同
            异常证据来源。
        """
        x = self.revin(x, "norm")
        z = x[:, : self.context_size, :]
        y = x[:, self.context_size :, :]
        time_loss_rec, time_loss_pre, time_adj = self.time_executor(z, y)
        wavelet_loss_rec, wavelet_loss_pre, wavelet_adj = self.wavelet_executor(z, y)

        time_rec = time_loss_rec.mean(dim=-1)
        time_pre = time_loss_pre.mean(dim=-1)
        wavelet_rec = wavelet_loss_rec.mean(dim=-1).mean(dim=-1)
        wavelet_pre = wavelet_loss_pre.mean(dim=-1).mean(dim=-1)

        # 图异常评分阶段：
        # 图相似度越低，说明时间域和时频域对当前窗口关联结构的解释越不一致；
        # 因此使用 `1 - similarity` 作为关联异常分数。
        cl = 1.0 - self.compute_graph_similarity(time_adj, wavelet_adj)
        if self.graph_window_size > 1:
            cl = self._window_to_patch_score(cl)

        # 多证据评分组合阶段：
        # 重构分数负责当前上下文偏离，预测分数负责未来动态偏离，图分数负责
        # 双域关联不一致。最终组合方式由 `wgad_score_mode` 控制。
        rec_hist = time_rec + self.score_lambda * wavelet_rec
        pre_future = time_pre + self.score_lambda * wavelet_pre

        time_rec_full = self._pad_right(time_rec, self.win_size)
        wavelet_rec_full = self._pad_right(wavelet_rec, self.win_size)
        rec_full = self._pad_right(rec_hist, self.win_size)
        time_pre_full = self._pad_left_with_median(time_pre, self.win_size)
        wavelet_pre_full = self._pad_left_with_median(wavelet_pre, self.win_size)
        pre_full = self._pad_left_with_median(pre_future, self.win_size)
        cl_full = self._patch_to_point_score(cl)

        score = self._compose_final_score(
            time_rec_score=time_rec_full,
            time_pre_score=time_pre_full,
            wavelet_rec_score=wavelet_rec_full,
            wavelet_pre_score=wavelet_pre_full,
            cl_score=cl_full,
            rec_score=rec_full,
            pre_score=pre_full,
        )
        return {
            "score": score,
            "time_rec": time_rec_full,
            "time_pre": time_pre_full,
            "wavelet_rec": wavelet_rec_full,
            "wavelet_pre": wavelet_pre_full,
            "cl": cl_full,
        }

    def _pad_right(self, score, target_len):
        if score.shape[1] >= target_len:
            return score[:, :target_len]
        pad_value = score[:, -1:]
        padding = pad_value.repeat(1, target_len - score.shape[1])
        return torch.cat([score, padding], dim=1)

    def _pad_left_with_median(self, score, target_len):
        if score.shape[1] >= target_len:
            return score[:, -target_len:]
        prefix = score.median(dim=1, keepdim=True).values.repeat(
            1, target_len - score.shape[1]
        )
        return torch.cat([prefix, score], dim=1)

    def _patch_to_point_score(self, patch_score):
        point_score = patch_score.repeat_interleave(self.patch_len, dim=1)
        return self._pad_right(point_score, self.win_size)

    def _window_to_patch_score(self, window_score):
        """将二级时空图窗口分数平均回 patch 级分数。

        时空图的图一致性分数定义在二级窗口上，而最终异常检测需要 patch/点级
        对齐。这里将每个窗口分数均匀分配给其覆盖的 patch，并对重叠贡献求均值。
        """
        if self.graph_window_size == 1:
            return window_score
        bs, window_count = window_score.shape
        patch_score = window_score.new_zeros(bs, self.patch_num)
        counts = window_score.new_zeros(1, self.patch_num)
        for win_idx in range(window_count):
            start = win_idx * self.graph_window_stride
            end = start + self.graph_window_size
            patch_score[:, start:end] += window_score[:, win_idx : win_idx + 1]
            counts[:, start:end] += 1
        return patch_score / counts.clamp_min(1.0)

    def _compose_final_score(
        self,
        *,
        time_rec_score,
        time_pre_score,
        wavelet_rec_score,
        wavelet_pre_score,
        cl_score,
        rec_score,
        pre_score,
    ):
        score_mode = str(self.score_mode).lower()
        eps = 1e-6
        if score_mode == "product":
            final_score = cl_score * rec_score * pre_score
        elif score_mode == "rec":
            final_score = rec_score
        elif score_mode == "pre":
            final_score = pre_score
        elif score_mode == "one_minus_cl":
            final_score = cl_score
        elif score_mode == "time_rec":
            final_score = time_rec_score
        elif score_mode == "rec_times_cl":
            final_score = rec_score * cl_score
        elif score_mode == "rec_over_pre":
            final_score = rec_score / (pre_score + eps)
        else:
            raise ValueError(f"Unsupported WGAD score_mode: {self.score_mode}")
        return final_score

    def compute_graph_similarity(self, adj_t, adj_w):
        """计算时间域图与时频域图的逐窗口余弦相似度。

        两个邻接矩阵被展平为图结构向量后做 L2 归一化。推理时使用
        `1 - similarity` 作为图不一致异常分数。
        """
        bs, patch_num, _, _ = adj_t.shape
        adj_t_flat = adj_t.view(bs, patch_num, -1)
        adj_w_flat = adj_w.view(bs, patch_num, -1)
        adj_t_norm = F.normalize(adj_t_flat, p=2, dim=-1)
        adj_w_norm = F.normalize(adj_w_flat, p=2, dim=-1)
        return (adj_t_norm * adj_w_norm).sum(dim=-1)

    def compute_graph_contrastive_loss(self, adj_t, adj_w):
        """双域图一致性对比学习目标。

        同一样本同一 patch/二级窗口下的时间域图和时频域图构成正样本对，
        batch 内其它图构成负样本。该损失约束正常模式下双域关联结构保持一致，
        使推理时的不一致程度可作为关联异常证据。
        """
        bs, patch_num, _, _ = adj_t.shape
        n = bs * patch_num
        adj_t_flat = adj_t.view(n, -1)
        adj_w_flat = adj_w.view(n, -1)
        adj_t_norm = F.normalize(adj_t_flat, p=2, dim=1)
        adj_w_norm = F.normalize(adj_w_flat, p=2, dim=1)
        logits = torch.matmul(adj_t_norm, adj_w_norm.T)
        labels = torch.arange(n, dtype=torch.long, device=adj_t.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

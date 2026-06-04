from abc import abstractmethod, ABCMeta
from enum import Flag
from typing import Tuple, Iterator
import os
from contextlib import contextmanager
import math
from enum import Enum, auto
from typing import Any
from typing import Optional
import torch
from torch import nn
from torch.nn import Module
from torch.autograd import Function
from torch import Tensor
from torch.nn import (
    Dropout, Linear, ModuleList,
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    BCEWithLogitsLoss as TorchBCEWithLogitsLoss,
    MSELoss as TorchMSELoss,
)
from layers.Diffusion_TS_Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


@contextmanager
def ensure_hf_cache(cache_dir: str) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in ("HF_HOME", "TRANSFORMERS_CACHE")}
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class PLUMESubModule(Module, metaclass=ABCMeta):
    """Base class for VAE's perturbator and classifier."""
    def __init__(self) -> None:
        super().__init__()
        self.loss: Tensor
        self.register_buffer("loss", torch.tensor(0.0))

    @abstractmethod
    def forward(  # pylint: disable=missing-param-doc
            self, inputs: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Forward pass implementation."""


class PoolingType(Enum):
    MEAN = auto()
    CLS = auto()


class Transformer(PLUMESubModule):
    def __init__(
            self,
            depth: int,
            emb_size: int,
            num_heads: int = 6,
            dropout: float = 0.1,
            pooling: PoolingType = PoolingType.MEAN,
            num_class_tokens: int = 1,
            positional_encoding: bool = False,
    ):
        super().__init__()
        self.dropout_ = dropout
        self.pooling = pooling
        self.num_heads = num_heads
        self.depth = depth
        self.num_class_tokens = num_class_tokens
        self.embedding_size: int = emb_size
        self.positional_encoding_ = positional_encoding
        self.dropout = Dropout(self.dropout_)
        self.positional_encoding: Optional[PositionalEncoding] = None
        if self.positional_encoding_:
            self.positional_encoding = PositionalEncoding(self.embedding_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.num_heads,
            dropout=self.dropout_,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=self.depth)
        self.classifier = Linear(self.embedding_size, 1)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        if self.positional_encoding:
            inputs = self.positional_encoding(inputs)
        encoded: Tensor = self.encoder(inputs)  # [B*2, W, D]
        encoded = self.dropout(encoded)  # [B*2, W, D]
        logits: Tensor = self.classifier(encoded)  # [B*2, W, 1]
        return logits.squeeze(-1)


class Perturbator(PLUMESubModule, metaclass=ABCMeta):
    """VAE perturbator."""


class PositionalEncoding(Module):
    def __init__(self, embedding_size, max_len: int = 5000, only_positions: bool = False):
        super().__init__()
        self.only_positions = only_positions
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.pe: Tensor

    def forward(self, inputs: Tensor) -> Tensor:
        if self.only_positions:
            return self.pe[:, :inputs.size(1)].repeat(inputs.size(0), 1, 1)
        return inputs + self.pe[:, :inputs.size(1)]


class BaseLoss(metaclass=ABCMeta):
    """Base class for the losses."""
    LOSS = "loss"
    def __init__(self, coefficient: float = 1.0) -> None:
        """Initialise the loss.
        Args:
            coefficient (float, optional): Coefficient for the loss. Defaults
                to 1.0.
        """
        self.coefficient = coefficient


class KLDivergence(BaseLoss):
    """Kullback-Leibler divergence used in VAEs."""
    LOSS = "kl_divergence"
    def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
        loss = (
                0.5
                * torch.sum(torch.exp(set2) + torch.pow(set1, 2) - 1.0 - set2)
                / set1.size(0)
        )
        return self.coefficient * loss


class MSELoss(BaseLoss):
    """Mean Square Error."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss = TorchMSELoss()

    def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
        loss = self._loss(set1, set2)
        return self.coefficient * loss


class ReSamplingStrategy(Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, samples: Tensor) -> Tensor:
        pass


class NoReSampling(ReSamplingStrategy):
    def forward(self, samples: Tensor) -> Tensor:
        return samples.detach()


class LogNormal(ReSamplingStrategy):
    def forward(self, samples: Tensor) -> Tensor:
        samples = samples.detach()
        return samples.sign() * samples.abs().exp()


class _Quantifier(metaclass=ABCMeta):
    """"""


class CoefficientScheduling(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.last_coefficient = None

    @abstractmethod
    def __call__(self, current_step: int, max_steps: int, coefficient: float) -> float:
        pass


class FixedCoefficient(CoefficientScheduling):
    def __call__(self, current_step: int, max_steps: int, coefficient: float) -> float:
        self.last_coefficient = coefficient
        return coefficient


class _ScaledQuantifier(_Quantifier, metaclass=ABCMeta):
    def __init__(self, coefficient: float = 1.0,
                 coefficient_scheduling: Optional[CoefficientScheduling] = None) -> None:
        self.coefficient_ = coefficient
        self.coefficient_scheduling = coefficient_scheduling or FixedCoefficient()

    @property
    def coefficient(self) -> float:
        return self.coefficient_scheduling(
            None, None,
            self.coefficient_,
        )


class BaseDispersion(_ScaledQuantifier, metaclass=ABCMeta):
    """Base dispersion class."""
    @abstractmethod
    def __call__(self, *, batch: Tensor) -> Tensor:
        """Calculate the dispersion between each sample in the batch.
        Returns:
            Tensor: The dispersion value.
        """


class EDTransformerVAE(Perturbator):
    class Projections(Flag):
        HIDDEN_TO_HIDDEN = auto()
        HIDDEN_TO_EMBEDDING = auto()
        OUTPUT = auto()
        NONE = auto()

    class PositionalEncoding_(Flag):
        SOURCE_LEVEL = auto()
        TARGET_LEVEL = auto()
        TARGET_LEVEL_POSONLY = auto()
        NONE = auto()

    class SamplingLevel(Enum):
        MEAN = auto()
        ALL = auto()

    class SecondaryTask(Enum):
        RECONSTRUCTION = auto()

    def __init__(
            self,
            parameters_loss: KLDivergence,
            reconstruction_loss: MSELoss,
            resampling_strategy: ReSamplingStrategy,
            depth: int,
            num_heads: int = 6,
            feature_size: int = 768,
            positional_encoding: PositionalEncoding_ = PositionalEncoding_.NONE,
            secondary_task: SecondaryTask = SecondaryTask.RECONSTRUCTION,
            dropout: float = 0.1,
            sampling_level: SamplingLevel = SamplingLevel.MEAN,
            sampling_constraint: Optional[BaseDispersion] = None,
            outputs_constraint: Optional[BaseDispersion] = None,
            residual: bool = False,
            forecasting_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.parameters_loss = parameters_loss
        self.reconstruction_loss = reconstruction_loss
        self.dropout = dropout
        self.positional_encoding_ = positional_encoding
        self.sampling_level = sampling_level
        self.sampling_constraint = sampling_constraint
        self.outputs_constraint = outputs_constraint
        self.residual = residual
        self.forecasting_length = forecasting_length
        self.secondary_task = secondary_task
        self.resampling_strategy = resampling_strategy
        self.positional_encoding: ModuleList = ModuleList([None, None])
        self.setup_with_provider_(feature_size=feature_size)

    def setup_with_provider_(self, feature_size: int) -> None:
        hidden_size = feature_size
        if self.PositionalEncoding_.SOURCE_LEVEL in self.positional_encoding_:
            self.positional_encoding[0] = PositionalEncoding(feature_size)
        encoder_layer = TransformerEncoderLayer(
            feature_size, self.num_heads, batch_first=True, dropout=self.dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=self.depth)
        self.encoder_mu = Linear(feature_size, hidden_size)
        self.encoder_sigma = Linear(feature_size, hidden_size)
        if self.PositionalEncoding_.TARGET_LEVEL in self.positional_encoding_:
            self.positional_encoding[1] = PositionalEncoding(hidden_size)
        elif self.PositionalEncoding_.TARGET_LEVEL_POSONLY in self.positional_encoding_:
            self.positional_encoding[1] = PositionalEncoding(hidden_size, only_positions=True)
        decoder_layer = TransformerDecoderLayer(
            hidden_size, self.num_heads, batch_first=True, dropout=self.dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=self.depth)
        decoder_layer = TransformerDecoderLayer(
            hidden_size, self.num_heads, batch_first=True, dropout=self.dropout
        )
        self.anomaly_decoder = TransformerDecoder(decoder_layer, num_layers=self.depth)

    def encode(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode the inputs to means and "variances" (`log(sigma**2)`).
        Arguments:
            inputs (Tensor): The input data.
        Returns:
            Tuple[Tensor, Tensor]: Mean and variances for each sample in the
                batch.
        """
        if self.positional_encoding[0]:
            inputs = self.positional_encoding[0](inputs)
        features: Tensor = self.encoder(inputs)
        if self.sampling_level == self.SamplingLevel.MEAN:
            features_1d = features.mean(dim=1)
        elif self.sampling_level == self.SamplingLevel.ALL:
            features_1d = features.reshape(-1, features.size(-1))
        mean = self.encoder_mu(features_1d)
        log_varsq = self.encoder_sigma(features_1d)
        return mean, log_varsq

    def reparametrize(self, mean: Tensor, log_varsq: Tensor) -> Tensor:
        """Reparametrisation trick, to backpropagate when using randomness.
        Arguments:
            mean (Tensor): Samples means.
            log_varsq (Tensor): Samples variances.
        Returns:
            Tensor: Sampled random vectors.
        """
        std = (log_varsq * 0.5).exp()
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, inputs: Tensor) -> Tensor:
        source_inputs = inputs.detach()
        mean, log_varsq = self.encode(source_inputs)
        sampled = self.reparametrize(mean, log_varsq)
        if isinstance(self.parameters_loss, KLDivergence):
            self.loss = self.parameters_loss(set1=mean, set2=log_varsq)
        else:
            self.loss = self.parameters_loss(
                set1=sampled, set2=torch.rand_like(sampled)
            )
        if self.sampling_constraint is not None:
            self.loss = self.loss - self.sampling_constraint(batch=sampled)
        if self.sampling_level == self.SamplingLevel.MEAN:
            sampled = sampled.unsqueeze(1).expand(-1, inputs.size(1), -1)
        elif self.sampling_level == self.SamplingLevel.ALL:
            sampled = sampled.reshape(*inputs.size())
        target_inputs = inputs.detach()
        if self.positional_encoding[1]:
            target_inputs = self.positional_encoding[1](target_inputs)
        reconstructed = self.decoder(target_inputs, sampled)
        if self.secondary_task is self.SecondaryTask.RECONSTRUCTION:
            self.loss = self.loss + self.reconstruction_loss(reconstructed, inputs)
        outputs = self.anomaly_decoder(target_inputs, self.resampling_strategy(sampled))
        if self.outputs_constraint is not None:
            self.loss = self.loss - self.outputs_constraint(batch=outputs)
        if self.residual:
            return inputs + outputs
        return outputs


class ActionModule(Module):
    pass


class StopGradient(ActionModule):
    class StopGradientFunction(Function):
        @staticmethod
        def forward(ctx: Any, input: Tensor) -> Tensor:
            return input

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tensor:
            return torch.zeros_like(grad_output)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.StopGradientFunction.apply(inputs)


class ReverseGradient(ActionModule):
    LOG_KEY = "reverse_gradient_coeff"
    def __init__(
            self,
            *args,
            max_epochs: int = -1,
            steps_per_epoch: int = -1,
            acc_grad: int = 1,
            warm_start: bool = False,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._acc_grad = acc_grad
        self._steps_per_epoch = math.ceil(steps_per_epoch / self._acc_grad)
        self._max_epochs = max_epochs
        self._warm_start = warm_start
        self._current_epoch = 0
        self._current_batch = -1
        self._current_update = -1
        self.coeff = 1.0

    class ReverseGradientFunction(Function):
        @staticmethod
        def forward(ctx: Any, input: Tensor, coeff: float) -> Tensor:
            ctx.coeff = coeff
            return input

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tensor:
            return -ctx.coeff * grad_output, None

    def _calculate_coeff(self) -> float:
        if self._max_epochs <= 0 or self._steps_per_epoch <= 0 or not self._warm_start:
            return self.coeff

        self._current_batch += 1
        if self._current_batch % self._acc_grad == 0:
            self._current_update += 1

        if self._current_batch >= self._steps_per_epoch:
            self._current_epoch += 1

        coeff_ = torch.tensor(
            self._current_update / (self._max_epochs * self._steps_per_epoch)
        )
        self.coeff = (2.0 / (1.0 + torch.exp(-10.0 * coeff_)) - 1.0).item()

        return self.coeff

    def reset(self) -> None:
        self._current_epoch = 0
        self._current_batch = -1
        self._current_update = -1

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ReverseGradientFunction.apply(
            inputs, self._calculate_coeff()
        )


class DummyModule(ActionModule):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class BCEWithLogitsLoss(BaseLoss):
    """Binary Cross-Entropy from logits."""
    LOSS = "bce"
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss = TorchBCEWithLogitsLoss()

    def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
        loss = self._loss(set1, set2)
        return self.coefficient * loss


class Generator_Classifier(nn.Module):
    """少量异常引导的双路径伪异常生成与统一判别模块。

    对应《二阶段技术方案》中的“少量异常引导的有监督微调”。该模块以
    MMask 编码后的 patch 级表征作为统一判别空间，并在训练阶段把正常
    表征、VAE 潜在空间伪异常表征、Diffusion-TS 时序空间伪异常表征拼接
    后送入同一个二分类器。VAE 分支强调生成靠近判别边界的困难伪异常；
    diffusion 分支强调在正常时序先验和可选引导函数下生成更接近真实异常
    形态的伪异常。推理阶段只保留分类器输出的异常 logit，作为分类分支
    异常评分。

    Args:
        window_size: 原始时间窗口长度，用于 diffusion 时序生成。
        num_features: patch 表征维度，即分类器输入的特征维度。
        signal_num: 原始序列通道数，用于 Diffusion-TS 的 feature_size。
        diffusion_cfg: Diffusion-TS 的结构和采样配置。
        use_gen_clf: 总开关；关闭时 VAE、diffusion 和分类器均不参与。
        use_vae: 是否启用潜在空间 VAE 对抗生成路径。
        use_diffusion: 是否启用时序空间 diffusion 引导生成路径。
        context_ratio: diffusion infill 采样时保留为上下文的时间比例。
    """

    def __init__(
            self,
            window_size: int,
            num_features: int,
            signal_num: int,
            diffusion_cfg: dict,
            use_gen_clf: bool = True,  # 总开关：是否启用异常生成+分类模块
            use_vae: bool = True,  # 子开关：启用VAE对抗生成
            use_diffusion: bool = True,  # 子开关：启用扩散引导生成
            context_ratio: float = 0.6,
            **kwargs
    ):
        super().__init__()
        self.use_gen_clf = use_gen_clf
        self.use_vae = use_vae
        self.use_diffusion = use_diffusion

        # 如果总开关关闭，则子开关强制失效
        if not self.use_gen_clf:
            self.use_vae = False
            self.use_diffusion = False

        self.window_size = window_size
        self.num_features = num_features
        self.signal_num = signal_num

        # --- 仅在总开关开启时初始化核心组件 ---
        if self.use_gen_clf:
            # 1. 分类器
            self.classifier = Transformer(
                emb_size=num_features,
                depth=2,
                pooling=PoolingType.CLS,
                positional_encoding=True,
                num_heads=4,
            )
            self.classifier_loss = BCEWithLogitsLoss()

            # 2. VAE 路径
            if self.use_vae:
                self.perturbator = EDTransformerVAE(
                    parameters_loss=KLDivergence(),
                    reconstruction_loss=MSELoss(),
                    resampling_strategy=NoReSampling(),
                    positional_encoding=EDTransformerVAE.PositionalEncoding_.SOURCE_LEVEL |
                                        EDTransformerVAE.PositionalEncoding_.TARGET_LEVEL_POSONLY,
                    depth=2, num_heads=4, feature_size=num_features,
                )
                self.stop_gradient = StopGradient()
                self.reverse_gradient = ReverseGradient()

            # 3. Diffusion 路径
            if self.use_diffusion:
                self.diffusion_ts = Diffusion_TS(
                    seq_length=window_size,
                    feature_size=signal_num,
                    **diffusion_cfg
                )
                self.context_ratio = context_ratio
                self.cond_fn = None

        self.patch_embed_meanc_fn = None

    def set_patch_embed_meanc_fn(self, fn):
        """注册原始时间序列到 patch 均值表征的投影函数。

        diffusion 分支在原始时序空间生成伪异常后，需要复用主干模型的
        patch 表征接口把生成序列映射回分类器输入空间。该注册方式避免
        diffusion 分支直接依赖 MMask 内部实现。
        """
        self.patch_embed_meanc_fn = fn

    def set_cond_fn(self, cond_fn):
        """注册 diffusion 反向采样的可微引导函数。

        技术方案中 diffusion 路径可由异常原型相似性和分类器判别反馈共同
        引导。这里的 `cond_fn` 是该引导项的接口；未设置时 diffusion 仅按
        自身 learned prior 做条件补全采样。
        """
        self.cond_fn = cond_fn

    def _adapt_targets(self, B, P, device, dtype):
        """生成统一判别训练标签。

        第一段 batch 是真实正常表征，标签为 0；后续 VAE 伪异常和
        diffusion 伪异常标签为 1。输出形状与分类器 patch 级 logits 对齐。
        """
        target_list = [torch.zeros((B, P), device=device, dtype=dtype)]
        if self.use_vae:
            target_list.append(torch.ones((B, P), device=device, dtype=dtype))
        if self.use_diffusion:
            target_list.append(torch.ones((B, P), device=device, dtype=dtype))
        return torch.cat(target_list, dim=0)

    def training_step(self, input_patch_meanc, ts):
        """执行一次生成增强分类训练前向。

        Args:
            input_patch_meanc: 主干编码后的正常 patch 表征，形状为
                `[B, P, D]`。
            ts: 原始时间窗口，形状为 `[B, T, C]`，供 diffusion 在时序空间
                进行 infill 伪异常生成。

        Returns:
            一个字典，包含分类 logits、生成与分类损失，以及 diffusion 伪异常
            的 patch 表征。`loss_gc` 是外部训练循环实际加入总 loss 的接口。
        """
        # 如果总开关关闭，直接返回零损失和空结果
        if not self.use_gen_clf:
            device = input_patch_meanc.device
            return {
                "predictions": None,
                "loss_gc": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_generator": torch.tensor(0.0, device=device),
                "loss_classifier": torch.tensor(0.0, device=device),
                "pseudo_anomalies": None
            }

        B, P, D = input_patch_meanc.shape
        device = input_patch_meanc.device

        classifier_inputs = [input_patch_meanc]
        total_gen_loss = torch.tensor(0.0, device=device)
        pseudo_diff_out = None

        if self.use_vae:
            # 潜在空间 VAE 对抗生成阶段：
            # 先阻断主干表征梯度，再由 VAE 生成伪异常表征；送入分类器前经过
            # 梯度反转，使分类器学习区分正常/伪异常，同时推动 VAE 生成更难
            # 区分的边界异常表征。
            pseudo_vae_emb = self.perturbator(self.stop_gradient(input_patch_meanc))
            classifier_inputs.append(self.reverse_gradient(pseudo_vae_emb))
            total_gen_loss += self.perturbator.loss

        if self.use_diffusion:
            # 时序空间 diffusion 引导生成阶段：
            # 保留窗口前段作为上下文，后段由 Diffusion-TS 条件补全生成。
            # 如果外部注册了 cond_fn，采样过程会额外受到异常原型或分类器梯度
            # 引导；生成后的时序伪异常再投影回 patch 表征空间参与统一判别。
            t_ctx = int(self.window_size * self.context_ratio)
            context_mask = torch.zeros_like(ts, dtype=torch.bool)
            context_mask[:, :t_ctx, :] = True
            target = torch.zeros_like(ts);
            target[context_mask] = ts[context_mask]

            ts_pseudo_diff = self.diffusion_ts.sample_infill(
                shape=ts.shape, target=target, partial_mask=context_mask,
                clip_denoised=True, cond_fn=self.cond_fn
            )
            pseudo_diff_emb = self.patch_embed_meanc_fn(ts_pseudo_diff)
            classifier_inputs.append(pseudo_diff_emb)
            pseudo_diff_out = pseudo_diff_emb

        # 统一监督判别阶段：
        # 将正常表征、VAE 伪异常表征和 diffusion 伪异常表征沿 batch 维拼接，
        # 用同一个二分类器学习正常/异常边界。该分类损失与生成损失一起作为
        # `loss_gc` 返回给外部训练循环。
        combined_input = torch.cat(classifier_inputs, dim=0)
        predictions = self.classifier(combined_input)
        targets = self._adapt_targets(B, P, device, input_patch_meanc.dtype)
        loss_classifier = self.classifier_loss(predictions, targets)

        return {
            "predictions": predictions,
            "loss_gc": loss_classifier + total_gen_loss,
            "loss_generator": total_gen_loss,
            "loss_classifier": loss_classifier,
            "pseudo_anomalies": pseudo_diff_out
        }

    def inference_step(self, inputs):
        """输出分类分支异常 logit。

        推理阶段不再生成伪异常，只把当前窗口的 patch 表征送入训练好的
        分类器。外部会对 logits 做 sigmoid 并插值到点级异常分数。
        """
        if not self.use_gen_clf:
            # 推理阶段如果关闭，返回全零分数
            return {"predictions": torch.zeros((inputs.shape[0], inputs.shape[1]), device=inputs.device)}

        predictions = self.classifier(inputs)
        return {"predictions": predictions}

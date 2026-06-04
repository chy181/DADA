import torch
import torch.nn as nn
import torch.nn.functional as F
# 从 SelfAttention_Family 引入 AttentionLayer 和 FullAttention
import math
import numpy as np

class Plugin(nn.Module):
    """跨尺度调制插件。

    对应《技术方案12-15》中的“多尺度检测与跨尺度调制”。多尺度外层会按
    不同下采样尺度分别构造窗口并调用同一个主干；该插件负责把粗粒度尺度
    得到的 patch 表征 `c` 作为上下文先验，生成细粒度表征的 AdaLN shift/scale
    调制量，从而在细粒度局部检测中注入宏观全局信息。

    该模块的线性输出层初始化为 0，因此刚启用时近似恒等残差，训练过程中再
    逐步学习跨尺度校准，避免破坏原主干表征。
    """

    def __init__(self, model_channels):
        super().__init__()

        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, model_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out output layers
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        """用粗尺度上下文调制当前尺度 patch 表征。

        Args:
            x: 当前尺度 patch 表征，通常形状为 `[B*C, P, D]`。
            c: 上一粗尺度缓存的 patch 表征，作为全局上下文先验。

        Returns:
            与 `x` 同形状的调制残差，外部会执行 `x + residual`。
        """
        # 上下文对齐阶段：
        # 粗尺度 batch 数通常小于细尺度 batch 数，这里按比例重复上下文，
        # 使每个细尺度窗口都能获得对应的粗尺度调制条件。
        repeats = x.shape[0] // c.shape[0]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = shift.repeat_interleave(repeats=repeats, dim=0)
        scale = scale.repeat_interleave(repeats=repeats, dim=0)

        # AdaLN 调制阶段：
        # 用粗尺度生成的 shift/scale 校准细尺度归一化表征，再通过零初始化
        # 线性层输出残差，实现跨尺度上下文引导。
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

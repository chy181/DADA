
import torch
from torch import nn
from einops import rearrange
from .moe import SoftMoE
import numpy as np
from transformers.activations import ACT2FN
from .star import STAR

class CosPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_id=60 ):
        super().__init__()
        pe = torch.zeros(max_id, embedding_dim)
        position = torch.arange(0, max_id, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda()

    def forward(self, inputs):
        # inputs 是你的ID张量
        return self.pe[inputs.long()]
    
class StatusEmbedding(nn.Module):
    """标识引导的状态变量编码层。

    对应《技术方案12-15》中的“状态感知的低秩适配器”。离散状态变量同时
    包含状态取值语义和变量身份语义：同一取值在不同阀门/泵/设备上可能具有
    不同物理含义。该模块将状态取值 ID 与通道 ID 拼接编码，再通过 SoftMoE
    从状态记忆库中选择专家组合，最后聚合为 patch 级状态 token，供 LoRA
    生成状态条件化适配参数。
    """

    def __init__(self, num_shared_experts, num_experts, seq_len, enc_in, K ,d_model,patch_len, stride):
        super().__init__()
        
        self.cpe = CosPositionalEncoding(d_model//2, max(len(enc_in)+1,sum(enc_in)) )
        self.soft_moe = SoftMoE(num_shared_experts, num_experts, seq_len, enc_in, K, hidden_size=d_model)
        self.temp_emb = nn.Linear(d_model*patch_len, d_model)
        self.patch_len = patch_len
        self.stride = stride
        self.star = STAR(d_model, d_model)
        
    def forward(self, x):
        b,l,n=x.shape
        # 标识联合编码阶段：
        # value_identy 表示离散状态取值，channel_identy 表示状态变量身份。
        # 二者拼接后可区分“哪个变量处于哪个状态”，避免只按数值类别建模。
        value_identy = self.cpe(rearrange(x, 'b l n  -> (b l) n'))
        channel_identy = self.cpe(torch.arange(x.size(-1))).unsqueeze(0).expand(value_identy.size(0),-1,-1)

        x = torch.cat((value_identy,channel_identy),dim=-1)

        # 状态记忆路由阶段：
        # SoftMoE 根据状态标识选择共享/专属专家的线性组合，并返回负载均衡
        # 损失，防止状态记忆集中使用少数专家。
        x, L_importance, weight = self.soft_moe(x)
        x = rearrange(x, '(b l) n d -> b l n d', b=b)

        # 点级到片段级聚合阶段：
        # 状态变量先按主干 patch_len/stride 切片，再映射成 patch 级状态 token，
        # 与连续变量 patch 表征在时间片段粒度对齐。
        x_patch = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_patch = rearrange(x_patch,'batch_size num_patch num_vars d_model patch_len -> batch_size num_vars num_patch (patch_len d_model)')
        
        # # 线性映射：变成 [B, L, d_model]
        status_token = self.temp_emb(x_patch)  # [B,n,num_patch, d_model]
        status_token = self.star(status_token)
        return x, status_token,  L_importance , weight

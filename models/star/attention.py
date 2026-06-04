
import torch
from torch import nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        batch_size, num_patch, d_model = query.shape
        x = x.unsqueeze(0).expand(batch_size,-1,-1)
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

from torch import nn
from einops import rearrange
from .embedding import  StatusEmbedding
from .star import  STAR
from .LoRA import LinearLoRA
import torch.nn.functional as F
import torch

class Plugin(nn.Module):
    """状态感知低秩适配器插件。

    该插件把离散状态建模接入 MMask patch 编码流程：`StatusEmbedding` 生成
    patch 级状态 token，`LinearLoRA` 根据状态 token 对连续变量 patch 表征
    做低秩残差适配，并通过状态/连续表征对比约束促使二者在同一片段语义上
    对齐。推理阶段输出状态相似度分数，可作为状态不一致异常证据。
    """

    def __init__(self, win_size, d_model, patch_len, stride, n_discrete, n_continuous, num_shared_experts, num_experts, discrete_nums, K, rank, alpha):
        super().__init__()

        self.n_discrete  = n_discrete
        self.n_continuous = n_continuous

        rank = rank
        self.status_embedding = StatusEmbedding(num_shared_experts=num_shared_experts, num_experts=num_experts, seq_len=win_size, enc_in=discrete_nums, K=K ,d_model=d_model,patch_len=patch_len, stride=stride)
        
        self.star_c = STAR(d_model, d_model)
        self.star_d = nn.Sequential(nn.Linear(d_model,d_model),nn.ReLU(),nn.Linear(d_model,d_model))
        self.status_token = None
        self.stride = stride
        self.patch_len = patch_len
        self.lora = LinearLoRA(in_features = patch_len, out_features=d_model, d_model=d_model, rank=rank, alpha=alpha, n_continuous = self.n_continuous)

    def forward(self, x_patch, discrete, emb_c):
  
        # 状态变量编码阶段：
        # 离散变量经过标识引导编码与状态记忆路由，得到 patch 级 discrete_token；
        # L_importance 是专家负载均衡约束，防止状态记忆使用塌缩。
        _, discrete_token, L_importance, weight0 = self.status_embedding(discrete)

        # 状态条件低秩适配阶段：
        # 用 discrete_token 生成 LoRA 调制参数，对连续变量原始 patch `x_patch`
        # 产生状态相关残差，再与主干连续表征 emb_c 相加。
        emb_c_lora = self.lora(x_patch, discrete_token)
        emb_c0 = emb_c_lora + emb_c
        emb_c1 = emb_c0

        # 状态-连续表征对齐阶段：
        # 连续表征经 STAR 聚合到变量维，离散状态 token 作为系统背景条件。
        # 对比损失约束同一片段下的状态表征与连续表征相互匹配。
        emb_d = discrete_token.squeeze(1).unsqueeze(2)
        emb_c = self.star_c(rearrange(emb_c0, '(b n) p d -> b n p d', n = self.n_continuous)).squeeze(1).unsqueeze(2)
        L_sim = contrastive_loss(emb_c,emb_d)*0.1
        return {"loss":L_importance + L_sim, "emb_c":emb_c1}
    
    def inference(self, x_patch, discrete, emb_c):

        # 推理阶段与训练阶段共享状态编码和低秩适配路径，但不计算负载/对比损失；
        # 额外输出连续表征与状态表征的余弦相似度作为状态一致性评分。
        _, discrete_token, L_importance, weight0 = self.status_embedding(discrete)
        emb_c_lora = self.lora(x_patch, discrete_token)
        emb_c0 = emb_c_lora + emb_c
        emb_c1 = emb_c0
        emb_d = discrete_token.squeeze(1).unsqueeze(2)
        emb_c = self.star_c(rearrange(emb_c0, '(b n) p d -> b n p d', n = self.n_continuous)).squeeze(1).unsqueeze(2)

        score = F.cosine_similarity(emb_c, emb_d, dim=-1).squeeze(-1)
        return {"score": score, "emb_c": emb_c1}
    
def contrastive_loss( emb_d, emb_c, temperature=0.07):

        emb_d = emb_d.mean(2)
        emb_c = emb_c.mean(2)
        b, p, d = emb_c.shape  
        
        queries = F.normalize(emb_d, p=2, dim=-1)
        keys = F.normalize(emb_c, p=2, dim=-1)
        logits = torch.matmul(queries, keys.transpose(-2, -1))
        logits = logits / temperature
        labels = torch.arange(p, device=logits.device).unsqueeze(0).expand(b, -1)
        return F.cross_entropy(logits.reshape(-1, p), labels.reshape(-1))

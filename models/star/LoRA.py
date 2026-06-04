import torch
from torch import nn
from einops import rearrange

class LinearLoRA(nn.Module):
    """状态感知的低秩适配线性层。

    对应《技术方案12-15》中的“状态感知的低秩适配”。基础低秩矩阵 A/B 保持
    冻结，状态 token 动态生成中间调制矩阵、幅值项和输入/输出瓶颈掩码。
    这样模型可以在不同系统状态下改变适配参数和有效秩，以低额外参数量刻画
    离散状态对连续数值模式的调节作用。
    """

    def __init__(self,
        in_features,
        out_features,
        d_model,
        rank,
        alpha,
        n_continuous=0):

        super().__init__()
 
        self.rank = out_features // rank
        self.alpha = alpha

        self.n_continuous = n_continuous
        self.n_vars = 1 

        # 低秩矩阵A和B
        self.lora_A = nn.Parameter(torch.zeros(in_features, self.rank),requires_grad=False)
        # 对A矩阵进行高斯初始化
        nn.init.kaiming_normal_(self.lora_A, a = 0.01)

        self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features),requires_grad=False)
        nn.init.kaiming_normal_(self.lora_B, a = 0.01)
        self.scale = self.alpha / self.rank

        self.r_embed=nn.Linear(d_model,self.rank*self.rank)
        self.b_embed=nn.Linear(d_model,out_features*1)
        self.mlp = nn.Sequential(nn.Linear(d_model,d_model), nn.ReLU(), nn.Linear(d_model, 2*self.rank+2))
        
    def forward(self, X, discrete_token, tau=0.001):        
        # self.name 
        original_shape = X.shape
        if len(original_shape)==3:
            X = rearrange(X,'(b n) p d -> b n p d', n=self.n_continuous)

        # X shape :[batch_size, n_vars, patch_num, in_Feature]
        # part 1
        batch_size=X.shape[0]
        self.n_vars=X.shape[1]
        input_patch_num=X.shape[2]

        status_token = discrete_token
        patch_num = status_token.shape[2]
        # print(self.name)
        # 去除prompt
        X = X[:,:,-patch_num:,:]

        # 状态条件参数生成阶段：
        # 由 patch 级状态 token 生成低秩中间矩阵 r、输出幅值 b，以及用于动态
        # 控制瓶颈尺寸的调制分数。
        r=self.r_embed(status_token) 
        b=self.b_embed(status_token) 
        R=r.shape[-1]
        r_s=int(pow(R,1/2))
        modulation_scores = self.mlp(status_token)

        # 动态输入瓶颈掩码：
        # 使用可微 sigmoid 门控近似二值掩码，使不同状态下参与低秩适配的输入
        # 子空间不同，对应方案中的“根据系统状态改变参数瓶颈尺寸”。
        input_mask_radio = modulation_scores[:,:,:,0].unsqueeze(-1)
        input_mask_score  = modulation_scores[:,:,:,1:self.rank+1] 
        input_mask = torch.sigmoid((input_mask_score - input_mask_radio) / tau).unsqueeze(-1)

        # 动态输出瓶颈掩码：
        # 与输入掩码配合控制 LoRA A/B 的有效秩，形成状态条件化的低秩更新。
        output_mask_radio = modulation_scores[:,:,:,self.rank+1].unsqueeze(-1)
        output_mask_score = modulation_scores[:,:,:,self.rank+1+1: 2*self.rank+1+1]
        output_mask = torch.sigmoid((output_mask_score - output_mask_radio) / tau).unsqueeze(-2)
 
        r = r.expand(-1,self.n_vars,-1,-1)
        b = b.expand(-1,self.n_vars,-1,-1)
        r=rearrange(r,'batch_size n_vars patch_num (r1 r2) -> (batch_size n_vars patch_num) r1 r2',r1=r_s,r2=r_s)
        b=rearrange(b,'batch_size n_vars patch_num d -> (batch_size n_vars patch_num) d 1')
        
        lora_A=self.lora_A.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size,self.n_vars,patch_num,-1,-1) #拓展维度
        lora_B=self.lora_B.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size,self.n_vars,patch_num,-1,-1)
        lora_A = lora_A*input_mask.transpose(-1,-2)
        lora_B = lora_B*output_mask.transpose(-1,-2)
        lora_A = rearrange(lora_A,'batch_size n_vars patch_num r1 r2 -> (batch_size n_vars patch_num) r1 r2')
        lora_B = rearrange(lora_B,'batch_size n_vars patch_num r1 r2 -> (batch_size n_vars patch_num) r1 r2')

        # 状态感知低秩权重合成阶段：
        # 冻结 A/B 提供基础低秩子空间，状态生成的 r 和 b 对该子空间做动态
        # 重参数化与幅值调制，得到每个样本、变量、patch 专属的适配权重。
        W = torch.einsum('bij,bjk,bkl->bil', lora_A, r, lora_B)  # [1536, 16, 32]
        # (batch_size*self.n_vars*patch_num) input output
        W = W*b.transpose(-1, -2)
 
        X = rearrange(X, 'b n patch_num in_Feature -> (b n patch_num) 1 in_Feature', n=self.n_vars)   # [B*, 1, in_Feature]
        
        X = torch.einsum('bij,bjk->bik',X, W)
        # X [batch_size, n_vars, patch_num, out_features]
        X = rearrange(X, '(b n patch_num) 1 out_features  -> b n patch_num out_features', n=self.n_vars, patch_num=patch_num)
        
        output = self.scale * X

        if len(original_shape)==3:
            output = rearrange(output,'b n p d -> (b n) p d', n=self.n_continuous)

        return output
    

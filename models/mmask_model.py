import torch
from einops import rearrange
from torch import nn
import numpy as np
from layers.dilated_conv import DilatedConvEncoder
from layers.adaptive_bottleneck import AdaptiveBottleNeck
from layers.gradient_reverse import WarmStartGradientReverseLayer
from layers.generator_classifier import Generator_Classifier
from models.star.Plugin import Plugin as STAR
from models.scale.Plugin import Plugin as MultiScale
from models.self_imp.Plugin import Plugin as SelfImp
from models.wgad.Plugin import Plugin as WGAD
import torch.nn.functional as F
import math
from math import sqrt


class MMaskModel(nn.Module):
    def __init__(
            self,
            win_size=100,
            patch_len=5,
            mask_mode="symmetry",
            hidden_dim=64,
            repr_dim=256,
            depth=10,
            adp_bottleneck=True,
            bottleneck_dims=[16, 32, 64, 128, 192, 256],
            k=3,
            revin=False,
            backbone="dilated_conv",
            max_iters=1e5,
            prototype="TC_res",
            n_channel=1,
            n_cluster=10,
            device='cpu',
            epsilon=0.05,
            temp_bern=0.07,
            classification=True,
            use_diffusion=True,
            use_vae=True,
            star=True,
            num_shared_experts=7,
            num_experts=10,
            K=5,
            rank=2,
            gama=0.5,
            n_continuous=None,
            n_discrete=None,
            discrete_nums=None,
    ):
        super().__init__()
        self.win_size = win_size
        # RevIn
        self.revin = revin
        # Patch
        self.patch_len = patch_len
        if win_size % patch_len:
            self.patch_num = (win_size // patch_len) + 1
        else:
            self.patch_num = win_size // patch_len
        # Encoder
        self.mask_mode = mask_mode
        self.hidden_dim = hidden_dim
        self.repr_dim = repr_dim
        self.n_channel = n_channel
        self.input_embed = nn.Linear(patch_len, hidden_dim)
        self.encoder = Encoder(
            patch_len=patch_len,
            patch_num=self.patch_num,
            output_dims=repr_dim,
            hidden_dims=hidden_dim,
            depth=depth,
            backbone=backbone,
        )
        # BottleNeck
        self.adp_bottleneck = adp_bottleneck
        if adp_bottleneck:
            self.adaptive_bottleneck = AdaptiveBottleNeck(
                seq_len=self.patch_num,
                seq_dim=repr_dim,  # update
                repr_dim=repr_dim,
                bn_dims=bottleneck_dims,
                k=k,
            )
        else:
            assert len(bottleneck_dims) == 1
            self.bottleneck = nn.Sequential(
                nn.Linear(repr_dim, bottleneck_dims[0]),
                nn.GELU(),
                nn.Linear(bottleneck_dims[0], bottleneck_dims[0]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(bottleneck_dims[0], repr_dim),
            )
        # Decoder
        self.decoder = MLP_Decoder(repr_dim, patch_len)
        self.adv_decoder = MLP_Decoder(repr_dim, patch_len)
        self.grl = WarmStartGradientReverseLayer(hi=0.5, max_iters=max_iters, auto_step=True)

        # 时间维度聚类模块：
        # 对应“残差特征聚类分配器”中的全局正常模式挖掘。聚类分配器把每个
        # patch 表征映射到聚类空间，并学习其对正常模式原型的软归属概率。
        # 推理时低归属置信度会转化为聚类相似度异常分数。
        self.prototype = prototype
        if self.prototype in ["TC", "TC_res"]:
            self.device = device
            self.epsilon = epsilon
            self.temp_bern = temp_bern
            self.n_channel = n_channel
            self.n_cluster = n_cluster
            self.Cluster_assigner = Cluster_assigner(n_vars=self.n_channel, n_cluster=self.n_cluster,
                                                     n_patches=self.patch_num, d_model=self.repr_dim * 2,
                                                     input_dim=hidden_dim * n_channel, device=self.device,
                                                     epsilon=self.epsilon, temp_bern=self.temp_bern)
            self.cluster_emb = self.Cluster_assigner.cluster_emb

        # 残差特征提取模块：
        # `TC_res` 在聚类前先构建样本级动态时间邻接图，用“当前 patch 表征 -
        # 邻域聚合表征”得到相对残差特征，从局部差异角度突出微弱异常偏离。
        if self.prototype in ["TC_res"]:
            self.graph_builder = TemporalGraphBuilder(seq_len=self.patch_num, feat_dim=hidden_dim * n_channel)
            self.neighbor_calculator = NeighborCalculator()
            self.residual_calculator = ResidualCalculator()

        self.win_size = win_size
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.star = False
        self.scale = False
        self.wgad = False
        self.self_imp = False

        # anomaly generation + classification
        self.classification = classification
        diffusion_cfg = dict(
            n_layer_enc=3,
            n_layer_dec=3,
            d_model=64,
            # timesteps=500,
            # sampling_timesteps=500,
            timesteps=5,
            sampling_timesteps=5,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.0,
            resid_pd=0.0,
            kernel_size=1,
            padding_size=0,
            context_ratio=0.6,
            guidance_scale=0.5,
            train_sampling_timesteps=None,
        )
        self.generator_classifier = Generator_Classifier(
            win_size, hidden_dim, n_channel, diffusion_cfg=diffusion_cfg,
            use_gen_clf=classification, use_vae=use_vae, use_diffusion=use_diffusion, context_ratio=0.6
        )

    def init_star(self, star, num_experts, num_shared_experts, K, rank, gama, n_continuous, n_discrete, discrete_nums):
        self.star = star
        self.star_module = STAR(win_size=self.win_size, d_model=self.hidden_dim, patch_len=self.patch_len,
                                stride=self.patch_len, num_experts=num_experts, num_shared_experts=num_shared_experts,
                                K=K, rank=rank, alpha=gama,
                                n_continuous=n_continuous, n_discrete=n_discrete, discrete_nums=discrete_nums)

    def init_ms(self, scales, ):
        self.scale = True
        self.scale_module = nn.ModuleList([MultiScale(model_channels=self.hidden_dim) for _ in range(1, len(scales))])

    def init_wgad(self, args):
        self.wgad = True
        self.wgad_module = WGAD(backbone=self, args=args, num_features=self.n_channel)

    def init_self_imp(self, args):
        self.self_imp = True
        self.self_imp_module = SelfImp(args=args)

    def forward(self, x, discrete, scale_index=0, grl=0):  # b x t x c
        self.scale_index = scale_index
        self.discrete = discrete
        x, _, cluster_prob, input_patch_flattenc, input_patch_meanc = self.encode(x, if_update=True)
        out, repr, balance_loss = self.decode(x, grl=grl)

        if self.star:
            if self.classification:
                return (out, repr, balance_loss, cluster_prob, input_patch_flattenc,
                        self.class_result['loss_gc'], self.class_result['loss_generator'],
                        self.class_result['loss_classifier'], self.star_result['loss'])
            else:
                return out, repr, balance_loss, cluster_prob, input_patch_flattenc, self.star_result['loss']
        else:
            if self.classification:
                return (out, repr, balance_loss, cluster_prob, input_patch_flattenc,
                        self.class_result['loss_gc'], self.class_result['loss_generator'],
                        self.class_result['loss_classifier'])
            else:
                return out, repr, balance_loss, cluster_prob, input_patch_flattenc

    def inference(self, x, discrete, scale_index=0, mask_mode=None, copies=10):
        self.scale_index = scale_index
        self.discrete = discrete
        if mask_mode is None:
            mask_mode = self.mask_mode
        B, T, dims = x.size()
        self.x = x
        input_patch, _, cluster_prob, input_patch_flattenc, input_patch_meanc = self.encode(x, if_update=False, inference=True)
        out = self.decode_inference(input_patch, mask_mode, copies)

        if self.star:
            if self.classification:
                return out, cluster_prob, self.class_result['predictions'], self.star_result['score']  # copies x b x t
            else:
                return out, cluster_prob, self.star_result['score']  # copies x b x t
        else:
            if self.classification:
                return out, cluster_prob, self.class_result['predictions']
            else:
                return out, cluster_prob

    def encode(self, x, if_update=True, inference=False):
        self.x = x
        # TEST OK
        B, T, dims = x.size()
        # 0.normalization
        if self.revin:
            self.means = x.mean(1, keepdim=True).detach()
            x = x - self.means
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= self.stdev
        # 1.channel independence
        x = x.permute(0, 2, 1)  # b x c x t
        x = x.reshape(B * dims, T)  # b*c x t
        # 2.do patch
        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            padding = torch.zeros([B, (length - T)]).to(x.device)
            input = torch.cat([x, padding], dim=1)  # b*c x patch_num*patch_len
        else:
            length = T
            input = x  # b*c x t
        patch = input.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # b*c x patch_num x patch_len
        input_patch = self.input_embed(patch)  # b*c x patch_num x hidden_dims
        input_patch_meanc = input_patch.reshape(B, -1, self.patch_num, input_patch.shape[2]).mean(
            dim=1)  # b x patch_num x hidden_dims

        if self.scale:
            if self.scale_index > 0:
                # 跨尺度调制阶段：
                # 粗尺度先处理并缓存 `last_input_patch`，细尺度编码时利用该粗尺度
                # 上下文生成调制残差，注入宏观趋势先验后再进入下游检测模块。
                input_patch = self.scale_module[self.scale_index - 1](input_patch, self.last_input_patch) + input_patch
            self.last_input_patch = input_patch

        # anomaly generation + classification
        if self.classification:
            if not inference:
                class_result_dict = self.generator_classifier.training_step(input_patch_meanc, ts=self.x)
            else:
                class_result_dict = self.generator_classifier.inference_step(input_patch_meanc)
            self.class_result = class_result_dict

        # 残差特征提取模块：
        # 先把所有通道的 patch 表征展平为时间维节点特征，构建每个样本专属的
        # 动态时间邻接矩阵；再聚合邻域表征并取残差，形成“局部差异”特征。
        input_patch_flattenc = input_patch.clone().reshape(B, self.patch_num, -1)  # b x patch_num x hidden_dims*c
        if self.prototype in ["TC_res"]:
            # 动态时间邻接构建：融合特征相似性和可学习时间距离先验。
            adjacency = self.graph_builder(input_patch_flattenc).to(x.device)
            # 邻域聚合：用样本级邻接矩阵加权聚合各时间 patch 的上下文模式。
            input_patch_flattenc = input_patch.clone().reshape(B, self.patch_num, -1,
                                                               self.hidden_dim)  # b x patch_num x c x hidden_dims
            node_embeddings = input_patch_flattenc
            neighbor_embeddings = self.neighbor_calculator(node_embeddings, adjacency)  # [b, win_size, c, hidden_dim]
            # 残差增强：正常 patch 与邻域模式接近，残差较小；异常 patch 偏离
            # 邻域动态模式，残差会被放大，更适合后续全局聚类分配。
            input_patch_flattenc = self.residual_calculator(node_embeddings,
                                                            neighbor_embeddings)  # [b, win_size, c, hidden_dim]
            input_patch_flattenc = input_patch_flattenc.reshape(B, self.patch_num, -1)  # b x patch_num x hidden_dims*c

        # 时间维度聚类模块：
        # 在原始 patch 特征或残差特征上进行端到端软聚类。输出的 cluster_prob
        # 是每个 patch 对各正常模式原型的归属概率，后续用于聚类损失和异常评分。
        if self.prototype in ["TC", "TC_res"]:
            cluster_prob, cluster_emb, cluster_mask = self.Cluster_assigner(input_patch_flattenc, self.cluster_emb)
        else:
            cluster_prob = None
        if if_update and self.prototype in ["TC", "TC_res"]:
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)

        if self.star:
            pass
            if not inference:
                # 状态感知低秩适配阶段：
                # STAR 根据离散状态变量生成状态 token，并由 LoRA 分支对连续
                # patch 表征做状态条件化调制，使数值模式在不同系统状态下拥有
                # 不同的异常判别基准。
                result_dict = self.star_module(x_patch=patch, discrete=self.discrete, emb_c=input_patch)
            else:
                result_dict = self.star_module.inference(x_patch=patch, discrete=self.discrete, emb_c=input_patch)
            input_patch = result_dict['emb_c']  # + input_patch
            self.star_result = result_dict

        return input_patch, patch, cluster_prob, input_patch_flattenc, input_patch_meanc

    def decode(self, input_patch, grl=0):
        B, T, dims = self.x.size()
        # 3.symmetry mask: generate copies
        if self.mask_mode == "symmetry":
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims, self.patch_num))).to(
                torch.bool)  # b*c x patch_num
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0)  # b*c*2 x patch_num
            input_patch = input_patch.repeat(2, 1, 1)
            input_patch[mask] = 0  # patch symmetry mask
        else:
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims, self.patch_num))).to(
                torch.bool)  # b*c x patch_num
            input_patch[mask] = 0
        # 4.encoder
        repr = self.encoder(input_patch)  # b*c*2 x patch_num x repr_dim
        # 5.adpBN
        balance_loss = torch.tensor(0., device=self.x.device, requires_grad=True)
        if self.adp_bottleneck:
            repr = torch.reshape(repr, (-1, self.patch_num * self.repr_dim))
            repr, balance_loss = self.adaptive_bottleneck(repr, repr)
            repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))
            repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))
        else:
            repr = self.bottleneck(repr)
        # 6.dual decoder
        if grl:
            gr_repr = self.grl(repr)
            out = self.adv_decoder(gr_repr)  # b*c*2 x patch_num*patch_len
        else:
            out = self.decoder(repr)  # b*c*2 x patch_num*patch_len
        # 7.symmetry mask: concat masked part
        if self.mask_mode == "symmetry":
            mask = mask.repeat(1, self.patch_len)  # b*c*2 x patch_num*patch_len
            out[~mask] = 0
            out = torch.reshape(out, (2, B * dims, self.patch_num * self.patch_len))  # 2 x b*c x patch_num*patch_len
            out = torch.sum(out, dim=0)  # b*c x patch_num*patch_len

        out = out[:, :T].reshape(B, dims, T)
        out = out.permute(0, 2, 1)  # b x t x c
        # de-Normalization
        if self.revin:
            out = out * self.stdev + self.means
        return out, repr, balance_loss

    def decode_inference(self, input_patch, mask_mode="Symmetry", copies=10, ):
        B, T, dims = self.x.size()
        # 3.mask: generate copies
        if mask_mode == "symmetry":
            assert copies % 2 == 0, "The number of copies of symmetric mask must be an even number"
            mask_1 = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims * (copies // 2), self.patch_num))).to(
                torch.bool)
            mask_2 = ~mask_1
            mask = torch.cat([mask_1, mask_2], dim=0)  # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0  # patch symmetry mask
        elif mask_mode == "random":
            mask = torch.from_numpy(np.random.binomial(1, 0.5, size=(B * dims * copies, self.patch_num))).to(
                torch.bool)  # b*c*copies x patch_num
            input_patch = input_patch.repeat(copies, 1, 1)
            input_patch[mask] = 0
        elif mask_mode == "nomask":
            copies = 1
        # 4.encoder
        repr = self.encoder(input_patch)  # b*c*copies x patch_num x repr_dim
        # 5.adpBN
        if self.adp_bottleneck:
            repr = torch.reshape(repr, (-1, self.patch_num * self.repr_dim))
            repr, balance_loss = self.adaptive_bottleneck(repr, repr)
            repr = torch.reshape(repr, (-1, self.patch_num, self.repr_dim))
        # 6.norm decoder
        out = self.decoder(repr)  # b*c*copies x patch_num*patch_len
        out = out.reshape(copies, B * dims, T)
        out = out[:, :, :T].reshape(copies, B, dims, T)
        out = out.permute(0, 1, 3, 2)  # copies x b x t x c
        # de-Normalization
        if self.revin:
            out = out * self.stdev.unsqueeze(dim=0).repeat(copies, 1, 1, 1) + self.means.unsqueeze(dim=0).repeat(copies,1,1,1)
        return out  # (copies) x b x t x c or (copies) x b x t

    def patch_embed_meanc(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: (B, win_size, C)
        return: (B, patch_num, hidden_dim)
        """
        B, T, C = ts.shape
        x = ts.permute(0, 2, 1).reshape(B * C, T)  # (B*C, T)
        if T % self.patch_len != 0:
            length = self.patch_num * self.patch_len
            pad_len = length - T
            padding = torch.zeros(B * C, pad_len, device=ts.device, dtype=ts.dtype)
            x = torch.cat([x, padding], dim=1)
        patch = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # (B*C, patch_num, patch_len)
        input_patch = self.input_embed(patch)  # (B*C, patch_num, hidden_dim)
        input_patch_meanc = input_patch.reshape(B, C, self.patch_num, input_patch.shape[-1]).mean(dim=1)
        return input_patch_meanc

    def patch_embed_channels(self, ts: torch.Tensor):
        """
        ts: (B, T, C)
        return:
            input_patch: (B*C, patch_num, hidden_dim)
            patch: (B*C, patch_num, patch_len)
        """
        B, T, C = ts.shape
        x = ts.permute(0, 2, 1).reshape(B * C, T)
        if T % self.patch_len != 0:
            patch_num = (T + self.patch_len - 1) // self.patch_len
            length = patch_num * self.patch_len
            pad_len = length - T
            padding = torch.zeros(B * C, pad_len, device=ts.device, dtype=ts.dtype)
            x = torch.cat([x, padding], dim=1)
        patch = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        input_patch = self.input_embed(patch)
        return input_patch, patch


class Encoder(nn.Module):
    def __init__(self, patch_len, patch_num, output_dims=512, hidden_dims=64, depth=10, backbone="dilated_conv"):
        super().__init__()
        self.patch_len = patch_len
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.backbone = backbone
        if backbone == "dilated_conv":
            self.feature_extractor = DilatedConvEncoder(
                hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
            )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # B x hidden_dims x patch_num
        repr = self.repr_dropout(self.feature_extractor(x))  # B x output_dims x patch_num
        repr = repr.permute(0, 2, 1)
        return repr  # B x patch_num x output_dims


class MLP_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        hidden_dim = int(input_dims * 2)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dims),
        )
        self.flatten = nn.Flatten(-2)

    def forward(self, x):  # b*c x patch_num x repr_dim
        x = self.net(x)  # b*c x patch_num x patch_len
        x = self.flatten(x)  # b*c x patch_num*patch_len
        return x


class Flatten_Decoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(-2),
            nn.Linear(input_dims, output_dims),
        )

    def forward(self, x):  # b*c x patch_num x repr_dim
        x = self.net(x)  # b*c x patch_num*patch_len
        return x


# 时间维度聚类模块
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):  # [n_patches, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True)
    Q = Q / (sum_Q)
    return Q


class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # scores = scores if mask == None else scores * mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A = A if mask == None else A * mask
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class CrossAttention(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)
    '''

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout=dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).repeat(B, 1, 1)  # [L,S] → [B,L,S]
            mask = mask.unsqueeze(1).repeat(1, H, 1, 1)  # [B,L,S] → [B,H,L,S]

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return out  # B, L, d_model


class Cluster_assigner(nn.Module):
    """端到端时间维度聚类分配器。

    对应《技术方案12-15》中的“时间维度聚类模块”。输入是每个样本的 patch
    特征或残差特征 `[B, P, D]`，模块先映射到专属聚类空间，再通过 Sinkhorn
    得到满足全局约束的软聚类分配。聚类中心随后由交叉注意力根据当前 batch
    的片段特征动态更新，实现特征学习与正常模式挖掘的协同优化。
    """

    def __init__(self, n_vars, n_cluster, n_patches, d_model, input_dim, device='cuda', epsilon=0.05, temp_bern=0.07):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_patches = n_patches
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.device = device
        self.linear = nn.Linear(input_dim, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        self.temp_bern = temp_bern

    def forward(self, x, cluster_emb):
        # x: [bs, seq_len, n_vars], [bs, n_patches, n_vars]
        # cluster_emb: [n_cluster, d_model]

        n_patches = x.shape[1]
        # 聚类空间映射与归一化阶段：
        # 将 patch/残差特征映射到聚类维度，并用 L2 归一化消除量纲差异，
        # 使相似度计算聚焦于模式方向而非特征幅值。
        x_emb = self.linear(x).reshape(-1, self.d_model)  # [bs*n_patches, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn / n_patches), 1)
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_patches, self.n_cluster)
        prob_temp = prob.reshape(-1, self.n_cluster)

        # 熵正则化软分配阶段：
        # Sinkhorn 在相似度矩阵上施加近似均衡约束，避免所有 patch 塌缩到少数
        # 原型，同时保留软概率形式以适应训练中特征分布的动态变化。
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)

        prob = prob_temp.reshape(bs, n_patches, self.n_cluster)  # [bs, n_patches, n_cluster]
        mask = self.concrete_bern(prob, temp=self.temp_bern)  # [bs, n_patches, n_cluster]

        # 注意力驱动的原型更新阶段：
        # 用可微 Concrete Bernoulli 掩码控制 patch-原型连接，再以原型为 query、
        # patch 特征为 key/value 做交叉注意力，使原型随当前 batch 正常模式更新。
        x_emb_ = x_emb.reshape(bs, n_patches, -1)
        cluster_emb_ = cluster_emb.repeat(bs, 1, 1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(1, 2))
        cluster_emb_avg = torch.mean(cluster_emb, dim=0)
        return prob, cluster_emb_avg, mask

    def concrete_bern(self, prob, temp=0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern


# 残差特征提取模块
class TemporalGraphBuilder(nn.Module):
    """动态时间邻接矩阵构建器。

    对应残差特征提取模块中的“双源融合邻接构建”。一方面根据当前样本 patch
    特征计算时间步间的内容相似性，另一方面保留可学习的时间距离先验；二者
    通过可学习系数融合，得到每个样本专属的时间邻接矩阵。
    """
    def __init__(self, seq_len=20, feat_dim=64, init_window=5, sparsity_threshold=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.init_window = init_window
        self.sparsity_threshold = sparsity_threshold
        self.learnable_adj = nn.Parameter(torch.randn(seq_len, seq_len))
        self.feat_weight = nn.Parameter(torch.ones(feat_dim))
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self._initialize_weights()

    def _initialize_weights(self):
        """原始的时间距离初始化"""
        with torch.no_grad():
            for i in range(self.seq_len):
                for j in range(self.seq_len):
                    dist = abs(i - j)
                    if dist <= self.init_window:
                        self.learnable_adj.data[i, j] = 1.0 / (dist + 1)
                    else:
                        self.learnable_adj.data[i, j] = -10.0

    def _compute_temporal_similarity(self, x):
        """计算批量样本的时间步特征相似性矩阵
        Args:
            x: 样本时序特征，形状 (batch_size, seq_len, feat_dim)
        Returns:
            sim_matrix: 每个样本的时间步相似性矩阵，形状 (batch_size, seq_len, seq_len)
        """
        # 特征相似性建模阶段：
        # 可学习特征权重会提升对异常敏感维度的贡献，再用余弦相似度刻画
        # patch 间的内容关联强度。
        x_weighted = x * self.feat_weight.unsqueeze(0).unsqueeze(0)  # (B, T, D) * (1,1,D) → (B,T,D)
        x_norm = F.normalize(x_weighted, p=2, dim=-1)  # (B, T, D)
        sim_matrix = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, T, T)
        sim_matrix = (sim_matrix + 1.0) / 2.0  # 归一化到 [0, 1]
        return sim_matrix

    def forward(self, x):
        """生成基于样本特征相似性的批量邻接矩阵
        Args:
            x: 样本时序特征，形状 (batch_size, seq_len, feat_dim)
        Returns:
            adj: 每个样本的专属时间邻接矩阵，形状 (batch_size, seq_len, seq_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        assert seq_len == self.seq_len, f"输入序列长度{seq_len}与初始化seq_len{self.seq_len}不匹配"
        sim_matrix = self._compute_temporal_similarity(x)  # (B, T, T)
        learnable_adj_batch = self.learnable_adj.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, T, T)
        fusion_alpha = torch.sigmoid(self.fusion_alpha)

        # 双源邻接融合阶段：
        # `sim_matrix` 表示样本内容驱动的动态关系，`learnable_adj` 表示时间距离
        # 先验。融合后归一化为邻域聚合权重。
        fused_adj = fusion_alpha * sim_matrix + (1 - fusion_alpha) * learnable_adj_batch  # (B, T, T)
        adj = F.softmax(fused_adj, dim=-1)
        adj = F.normalize(adj, p=1, dim=-1)
        return adj


class ResidualCalculator(nn.Module):
    """差异增强残差特征计算器。

    将绝对 patch 表征转换为相对差异表征：正常片段通常可由邻域模式解释，
    因而残差较小；异常片段偏离局部动态关系，残差更突出。
    """
    def __init__(self):
        super(ResidualCalculator, self).__init__()

    def forward(self, node_embeddings, neighbor_embeddings):
        """
        Args:
            node_embeddings: [b, win_size, c, hidden_dim]
            neighbor_embeddings: [b, win_size, c, hidden_dim]
        Returns:
            [b, win_size, c, hidden_dim]
        """
        residual_features = node_embeddings - neighbor_embeddings
        return residual_features


class NeighborCalculator(nn.Module):
    """基于动态时间邻接的邻域表征聚合器。"""
    def __init__(self):
        super(NeighborCalculator, self).__init__()

    def forward(self, node_embeddings, adjacency):
        """
        Args:
            node_embeddings: [b, win_size, c, hidden_dim]
            adjacency: [b, win_size, win_size]
        Returns:
            [b, win_size, c, hidden_dim]
        """
        B, T, C, D = node_embeddings.shape
        node_flat = node_embeddings.reshape(B, T, -1)  # [b, T, c*D]
        neighbor_flat = torch.bmm(adjacency, node_flat)  # [b, T, c*D]
        neighbor_embeddings = neighbor_flat.reshape(B, T, C, D)  # [b, T, c, D]
        return neighbor_embeddings

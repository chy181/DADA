import torch.nn as nn
from einops import rearrange
import torch
from utils.fre_rec_loss import frequency_criterion
from .Plugin import Plugin


class Model(nn.Module):
    """多尺度窗口检测与分数融合外壳。

    对应《技术方案12-15》中的“多尺度检测与跨尺度调制”。该类不改变底层
    MMask 主干接口，而是在外层按多个尺度对原始序列进行平均下采样、切分为
    固定 `win_size=100` 的局部窗口，并逐尺度调用主干获得重构、频域、聚类、
    分类、WGAD 和 SelfImp 等分数。各尺度分数会插值回原时间分辨率并相乘，
    用于同时覆盖短时突变和长跨度异常。
    """

    def __init__(self, args, backbone):
        super(Model, self).__init__()
        backbone_dim = 64
        patch_len = 5
        self.device = 'cuda'
        self.args = args
        self.backbone_dim = backbone_dim
        self.backbone_patch_len = patch_len
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        # self.freeze_backbone()

        if self.args.multi_scale in ['multi']:
            self.scales = self.args.scales
            self.backbone.init_ms(self.scales)
        else:
            self.scales = [1]
        if self.args.wgad:
            self.backbone.init_wgad(self.args)
        if getattr(self.args, "self_imp", False):
            self.backbone.init_self_imp(self.args)

    def freeze_backbone(self, ):
        self.backbone.load_state_dict(torch.load('temp.pt'))
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

    def unfreeze_backbone(self, ):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def inference(self, x, discrete):
        b, t, c = x.shape
        x0 = x.permute(0, 2, 1)
        state0 = discrete.permute(0, 2, 1)
        score_total = 1
        for i, scale in enumerate(self.scales[::-1]):
            # 多尺度重采样阶段：
            # 对连续变量在时间维按当前 scale 做平均下采样。这样单个 100 点
            # 主干窗口在粗尺度下可覆盖更长真实时间跨度，突破固定窗口长度限制。
            x = x0.unfold(dimension=-1, size=scale, step=scale).mean(-1)

            # 固定窗口切分阶段：
            # 下采样后的序列统一切成 100 点窗口，保持主干模型输入接口不变。
            x = x.unfold(dimension=-1, size=100, step=100)
            _, _, n_win, _ = x.shape
            x_in = rearrange(x, 'b n_var n_win win_size -> (b n_win) win_size n_var')

            if self.args.star:
                # 离散状态同步重采样阶段：
                # 状态变量是类别值，不能平均；这里用 mode 保留每个 scale 段内
                # 最常见状态，使状态感知适配器与连续变量窗口处于同一尺度。
                state = state0.unfold(dimension=-1, size=scale, step=scale).mode(-1).values
                state = state.unfold(dimension=-1, size=100, step=100)
                state_in = rearrange(state, 'b n_var n_win win_size ->(b n_win) win_size n_var')
                if self.args.classification:
                    out_copies, cluster_prob, predictions_class, score_star = self.backbone.inference(x_in, state_in,
                                                                                                      mask_mode='symmetry',
                                                                                                      copies=10,
                                                                                                      scale_index=i)
                else:
                    out_copies, cluster_prob, score_star = self.backbone.inference(x_in, state_in, mask_mode='symmetry',
                                                                                   copies=10, scale_index=i)
                score_star = nn.functional.interpolate(score_star.unsqueeze(1), size=100, mode='linear').squeeze(1)
                if i != len(self.scales) - 1:
                    score_star = nn.functional.interpolate(score_star.unsqueeze(1), size=100 * scale,
                                                           mode='linear').squeeze(1)
                score_star = rearrange(score_star, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
            else:
                if self.args.classification:
                    out_copies, cluster_prob, predictions_class = self.backbone.inference(x_in, state0,
                                                                                          mask_mode='symmetry',
                                                                                          copies=10, scale_index=i)
                else:
                    out_copies, cluster_prob = self.backbone.inference(x_in, state0, mask_mode='symmetry', copies=10,
                                                                       scale_index=i)

            # 当前尺度基础分数阶段：
            # 重构误差刻画时域还原偏差，频域误差刻画谱结构偏差。二者先在当前
            # 尺度局部窗口内计算，再插值回原尺度长度用于跨尺度融合。
            score_rec_local = ((out_copies.mean(0) - x_in) ** 2).mean(-1)
            if i != len(self.scales) - 1:
                score_rec = nn.functional.interpolate(score_rec_local.unsqueeze(1), size=100 * scale, mode='linear').squeeze(
                    1)
            else:
                score_rec = score_rec_local
            score_rec = rearrange(score_rec, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
            freq_anomaly_criterion = frequency_criterion(self.args)
            score_freq_local = (freq_anomaly_criterion(x_in, out_copies.mean(0))).mean(-1)
            if i != len(self.scales) - 1:
                score_freq = nn.functional.interpolate(score_freq_local.unsqueeze(1), size=100 * scale,
                                                       mode='linear').squeeze(1)
            else:
                score_freq = score_freq_local
            score_freq = rearrange(score_freq, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
            score_main = score_rec + self.args.score_lambda * score_freq
            score_channel_list = [score_rec_local, score_freq_local]
            if self.args.wgad:
                wgad_result = self.backbone.wgad_module.inference(x_in)
                score_wgad_local = wgad_result["score"]
                score_channel_list.append(score_wgad_local)
                if i != len(self.scales) - 1:
                    score_wgad = nn.functional.interpolate(
                        score_wgad_local.unsqueeze(1), size=100 * scale, mode='linear'
                    ).squeeze(1)
                else:
                    score_wgad = score_wgad_local
                score_wgad = rearrange(score_wgad, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
                score_main = self._blend_scores(score_main, score_wgad, self.args.wgad_alpha)
            if getattr(self.args, "self_imp", False):
                score_channels = torch.stack(score_channel_list, dim=-1)
                score_self_imp = self.backbone.self_imp_module.inference(x_in, score_channels)["score"]
                if i != len(self.scales) - 1:
                    score_self_imp = nn.functional.interpolate(
                        score_self_imp.unsqueeze(1), size=100 * scale, mode='linear'
                    ).squeeze(1)
                score_self_imp = rearrange(score_self_imp, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
                score_main = self._blend_scores(score_main, score_self_imp, self.args.self_imp_alpha)
            score_rec = score_main

            # compute similarity anomaly score
            if self.args.prototype in ['TC', 'TC_res']:
                # 聚类相似度分数阶段：
                # 聚类分配最大概率越低，说明该 patch 越难归入已学习的全局正常
                # 模式原型。该分数与重构/频域主分数按幂次乘性融合。
                cluster_prob_max, _ = torch.max(cluster_prob, dim=-1)  # [bs*n_win, n_patches]
                score_sim = 1 - cluster_prob_max
                score_sim = nn.functional.interpolate(score_sim.unsqueeze(1), size=100, mode='linear').squeeze(
                    1)  # [bs*n_win, win_size]
                if i != len(self.scales) - 1:
                    score_sim = nn.functional.interpolate(score_sim.unsqueeze(1), size=100 * scale,
                                                          mode='linear').squeeze(1)
                score_sim = rearrange(score_sim, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
                score_rec = torch.pow(score_rec, 1.0 - self.args.alpha1)
                score_sim = torch.pow(score_sim, self.args.alpha1)
                score = score_rec * score_sim
            else:
                score = score_rec

            # compute classification anomaly score
            if self.args.classification and i == 0:
                score_cla = torch.sigmoid(predictions_class)
                score_cla = nn.functional.interpolate(score_cla.unsqueeze(1), size=100, mode='linear').squeeze(
                    1)  # [bs*n_win, win_size]
                if i != len(self.scales) - 1:
                    score_cla = nn.functional.interpolate(score_cla.unsqueeze(1), size=100 * scale,
                                                          mode='linear').squeeze(1)
                score_cla = rearrange(score_cla, '(b n_win) win_size ->b (n_win win_size)', n_win=n_win)
                score = self._fuse_scores(score_rec, score_cla).detach().cpu().numpy()
            else:
                score = score.detach().cpu().numpy()

            # 跨尺度分数聚合阶段：
            # 不同尺度从不同时间分辨率观察异常，使用乘性聚合强化多尺度共同
            # 响应的位置，同时保留长跨度与短跨度异常的互补证据。
            score_total = score_total * score
        return score_total

    def train_function(self, x, discrete):
        b, t, c = x.shape
        x0 = x.permute(0, 2, 1)
        state0 = discrete.permute(0, 2, 1)
        loss_total = 0
        loss_rec = 0
        loss_s = 0
        loss_gc = 0
        loss_generator = 0
        loss_classifier = 0
        for i, scale in enumerate(self.scales[::-1]):
            # 训练阶段沿用与推理一致的多尺度窗口构造，保证每个尺度的重构、
            # 聚类和状态适配损失都作用在相同的尺度视图上。
            x = x0.unfold(dimension=-1, size=scale, step=scale).mean(-1)
            x = x.unfold(dimension=-1, size=100, step=100)
            _, _, n_win, _ = x.shape
            x_in = rearrange(x, 'b n_var n_win win_size ->(b n_win) win_size n_var')

            if self.args.star:
                # 离散状态训练对齐阶段：
                # 与连续变量同尺度切窗，但用众数保持类别语义，供 STAR 生成
                # 状态 token 与低秩适配参数。
                state = state0.unfold(dimension=-1, size=scale, step=scale).mode(-1).values
                state = state.unfold(dimension=-1, size=100, step=100)
                state_in = rearrange(state, 'b n_var n_win win_size ->(b n_win) win_size n_var')
                if self.args.classification:
                    x_out, repr, balance_loss, cluster_prob, input_patch_flattenc, loss_gc, loss_generator, loss_classifier, loss_star = self.backbone(
                        x_in, state_in, scale_index=i)
                else:
                    x_out, repr, balance_loss, cluster_prob, input_patch_flattenc, loss_star = self.backbone(
                        x_in, state_in, scale_index=i)
            else:
                if self.args.classification:
                    x_out, repr, balance_loss, cluster_prob, input_patch_flattenc, loss_gc, loss_generator, loss_classifier = self.backbone(
                        x_in, discrete, scale_index=i)
                else:
                    x_out, repr, balance_loss, cluster_prob, input_patch_flattenc = self.backbone(
                        x_in, discrete, scale_index=i)

            loss_rec = ((x_in - x_out) ** 2).mean()
            if self.args.wgad:
                loss = self.args.wgad_loss_weight * self.backbone.wgad_module(x_in)["loss"]
            else:
                loss = torch.tensor(0.0, device=x_in.device)

            # compute similarity loss
            if self.args.prototype in ['TC', 'TC_res']:
                simMatrix = self.get_similarity_matrix_update(input_patch_flattenc)
                loss_s = self.similarity_loss_batch(cluster_prob, simMatrix)
                loss = loss + loss_rec + self.args.beta1 * loss_s
            else:
                loss_s = torch.tensor(0).to(self.device)
                loss = loss + loss_rec

            # compute classification loss
            if self.args.classification and i == 0:
                loss = loss + self.args.beta2 * loss_gc
            elif self.args.classification and i != 0:
                loss_gc = loss_gc
                loss_generator = loss_generator
                loss_classifier = loss_classifier
            else:
                loss_gc = torch.tensor(0).to(self.device)
                loss_generator = torch.tensor(0).to(self.device)
                loss_classifier = torch.tensor(0).to(self.device)

            loss_total = loss_total + loss
        return loss_total, loss_rec, loss_s, loss_gc, loss_generator, loss_classifier

    def get_similarity_matrix_update(self, batch_data, sigma=5.0):
        x = batch_data  # [b, t, c]
        cdist = torch.cdist(x, x, p=2.0)  # [b, t, t]
        sim_matrix = torch.exp(-cdist / (2 * sigma ** 2))  # [b, t, t]
        return sim_matrix.to(self.device)

    def similarity_loss_batch(self, prob, simMatrix):
        def concrete_bern(prob, temp=0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern
        membership = concrete_bern(prob)  # [n_channels, n_clusters]
        S = simMatrix  # [b, t, t]
        M = membership.to(device=S.device, dtype=S.dtype)
        M_permuted = M.permute(0, 2, 1)
        temp_1 = torch.bmm(M_permuted, S)
        SAS = torch.bmm(temp_1, M)
        M_MT = torch.bmm(M, M_permuted)
        ones = torch.ones_like(M_MT, device=S.device, dtype=S.dtype)
        _SS = ones - M_MT  # [bs, n_patches, n_patches]
        SS = torch.bmm(_SS, S)
        trace_SAS = torch.diagonal(SAS, dim1=1, dim2=2).sum(dim=1)
        trace_SS = torch.diagonal(SS, dim1=1, dim2=2).sum(dim=1)
        term3 = M.shape[1] * torch.ones_like(trace_SAS, device=S.device)  # M.shape[1] = n_patches
        loss = -trace_SAS.sum() + trace_SS.sum() + term3.sum()
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss

    def _fuse_scores(self, score_rec, score_cla):
        strategy = self.args.weight_strategy.lower()
        alpha2 = self.args.alpha2
        if strategy == "fixed":
            fused_score = alpha2 * score_cla + (1.0 - alpha2) * score_rec
        elif strategy == "softmax":
            weight = torch.softmax(torch.stack([score_cla, score_rec]), dim=0)
            fused_score = weight[0] * score_cla + weight[1] * score_rec
        else:
            fused_score = alpha2 * score_cla + (1.0 - alpha2) * score_rec
        return fused_score

    def _blend_scores(self, score_main, score_aux, alpha):
        strategy = self.args.weight_strategy_blend.lower()
        if strategy == "fixed":
            return alpha * score_aux + (1.0 - alpha) * score_main
        if strategy == "softmax":
            weight = torch.softmax(torch.stack([score_aux, score_main]), dim=0)
            return weight[0] * score_aux + weight[1] * score_main
        return alpha * score_aux + (1.0 - alpha) * score_main

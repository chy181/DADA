import torch
import torch.nn.functional as F


def extract_anomaly_prototypes_from_test_loader(
    test_loader,
    patch_embed_meanc_fn,
    window_size: int,
    stride: int = 1,
    top_k: int = 50,
    anomaly_window_usage_ratio: float = 1.0,
    device: str = "cuda"
):
    original_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        original_cuda_rng_state = torch.cuda.get_rng_state()

    try:
        raw_anomaly_windows = []
        anomaly_embeddings = []

        for idx, (batch_x, discrete, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            for sample_idx in range(batch_x.shape[0]):
                sample = batch_x[sample_idx : sample_idx+1]
                label  = batch_y[sample_idx]

                anomaly_timesteps = torch.where(label == 1)[0].cpu().numpy()
                if len(anomaly_timesteps) == 0: continue

                for t in anomaly_timesteps:
                    start = max(0, t - window_size // 2)
                    end = min(sample.shape[1], start + window_size)
                    start = max(0, end - window_size)
                    anomaly_window = sample[:, start:end, :]
                    raw_anomaly_windows.append(anomaly_window.cpu())

                    with torch.no_grad():
                        patch_meanc = patch_embed_meanc_fn(anomaly_window)
                        window_embed = patch_meanc.mean(dim=1)
                    anomaly_embeddings.append(window_embed.cpu())

        if len(anomaly_embeddings) == 0:
            raise ValueError("未找到异常窗口")

        total_anomaly_window = len(anomaly_embeddings)
        usage_num = max(1, int(total_anomaly_window * anomaly_window_usage_ratio))
        idx_shuffle = torch.randperm(total_anomaly_window)[:usage_num]

        anomaly_embeddings = [anomaly_embeddings[i] for i in idx_shuffle]
        raw_anomaly_windows = [raw_anomaly_windows[i] for i in idx_shuffle]

        anomaly_embeddings = torch.cat(anomaly_embeddings, dim=0)
        unique_embeddings = []
        for emb in anomaly_embeddings:
            if all(torch.norm(emb - u) > 1e-6 for u in unique_embeddings):
                unique_embeddings.append(emb)
        unique_embeddings = torch.stack(unique_embeddings)

        if len(unique_embeddings) > top_k:
            select_idx = torch.randperm(len(unique_embeddings))[:top_k]
            anomaly_prototypes = unique_embeddings[select_idx]
        else:
            anomaly_prototypes = unique_embeddings

        return anomaly_prototypes.to(device), raw_anomaly_windows

    finally:
        torch.set_rng_state(original_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(original_cuda_rng_state)


def build_patch_level_anomaly_cond_fn(
    patch_embed_meanc_fn,
    patch_classifier_fn,
    guidance_scale: float,
    patch_weight=None,
    use_cls_gradient=True,        # 分类器梯度开关
    use_proto_gradient=True,     # 原型梯度开关
    anomaly_proto=None,
    proto_temperature=0.07,
    proto_weight=1.0,
):
    def cond_fn(x, t, **model_kwargs):
        x_in = x.detach().requires_grad_(True)
        total_loss = 0

        if use_cls_gradient:
            patch_meanc = patch_embed_meanc_fn(x_in)
            logits = patch_classifier_fn(patch_meanc)
            prob = torch.sigmoid(logits)

            if patch_weight is not None:
                w = patch_weight.to(prob.device).view(1, -1)
                R_cls = (prob * w).sum(dim=1) / (w.sum() + 1e-9)
            else:
                R_cls = prob.mean(dim=1)

            total_loss += R_cls.sum()

        if use_proto_gradient and anomaly_proto is not None:
            patch_meanc = patch_embed_meanc_fn(x_in)
            xg = F.normalize(patch_meanc.mean(dim=1), dim=-1)
            proto = anomaly_proto

            if proto.dim() == 3:
                proto = proto.mean(dim=1)
            proto = F.normalize(proto, dim=-1)

            sim = xg @ proto.T
            w = torch.softmax(sim / proto_temperature, dim=-1)
            R_proto = (w * sim).sum(dim=-1)
            total_loss += R_proto.sum() * proto_weight

        grad = torch.autograd.grad(
            outputs=total_loss,
            inputs=x_in,
            retain_graph=False,
            create_graph=False
        )[0]

        return grad * guidance_scale

    if use_proto_gradient and anomaly_proto is None:
        raise ValueError("启用原型梯度必须传入 anomaly_proto！")

    return cond_fn

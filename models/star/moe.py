import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, num_experts, encoder_hidden_size):
        super(MLP, self).__init__()
        self.distribution_fit = nn.Sequential(
                nn.Linear(encoder_hidden_size, encoder_hidden_size, bias=False),
                nn.ReLU(),
                nn.Linear(encoder_hidden_size, num_experts, bias=False)
            )
    def forward(self, x):
        out = self.distribution_fit(x)
        return out
    
class SoftMoE(nn.Module):
    def __init__(self, num_shared_experts, num_experts, seq_len, enc_in, K, hidden_size=32, noisy_gating=True, dropout = 0.2):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.input_size = seq_len
        self.K = K
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.Embedding(self.num_shared_experts + self.num_experts,hidden_size)
        self.W_h = nn.Parameter(torch.eye(self.num_shared_experts +self.num_experts)) 
        self.softplus = nn.Softplus()
        self.gate = MLP(self.num_shared_experts + num_experts, hidden_size) 
        self.noise = MLP(self.num_shared_experts + num_experts, hidden_size)
        self.n_vars = enc_in
        assert self.K <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, sigmoid_epsilon=1e-3):
        
        x = self.dropout(x)
        clean_logits = self.gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        shared_weight = torch.softmax(logits[...,:self.num_shared_experts],dim=-1)
        logits = logits[...,-self.num_experts:]

        top_logits, top_indices = logits.topk(min(self.K, self.num_experts), dim=-1)
        
        threshold = top_logits[...,-1].unsqueeze(-1)
        selection = torch.sigmoid((logits-threshold)/sigmoid_epsilon)
        selection = torch.sigmoid((selection-0.6)/sigmoid_epsilon)
        inf = torch.log(selection+1e-5)
        weight = torch.softmax(logits + inf, dim=-1)

        importance_cv = self.cv_squared(weight.sum(1).sum(0)) + self.cv_squared(shared_weight.sum(1).sum(0))
        load_cv = self.cv_squared(selection.sum(1).sum(0))
        weight0 = weight.clone()
        weight = torch.cat((shared_weight,weight),dim=-1) 
        return weight, importance_cv+load_cv, weight0
    
    def forward(self, x, loss_coef=1):
        weight, cv_loss, weight0 = self.noisy_top_k_gating(x, self.training)
        x = weight@self.experts.weight
        cv_loss *= loss_coef
        
        return x, cv_loss, weight0

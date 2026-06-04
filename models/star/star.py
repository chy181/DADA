import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

class STAR(nn.Module):
    def __init__(self, d_model_in, d_model_out):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """
        
        self.gen1 = nn.Linear(d_model_in , d_model_in)
        self.gen2 = nn.Linear(d_model_in, d_model_in)
        self.gen3 = nn.Linear(d_model_in, d_model_out)
        self.gen4 = nn.Linear(d_model_out, d_model_out)

    def forward(self, input):
        batch_size, channels, num_patch, d_model = input.shape
        combined_mean = rearrange(input,'b n p d -> (b p) n d')
        combined_mean = F.gelu(self.gen1(combined_mean))
        combined_mean = self.gen2(combined_mean)
        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size*num_patch, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True)
        combined_mean = F.gelu(self.gen3(combined_mean))
        combined_mean = self.gen4(combined_mean)
        return rearrange(combined_mean,'(b p) 1 d -> b 1 p d',p=num_patch)
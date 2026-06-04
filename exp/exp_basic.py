import os
import torch
from data_provider.data_provider import data_provider
from models import *

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MMask': MMaskModel,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.continuous_cols = None
        self.discrete_cols = None

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, discrete=True,):
        data_set, data_loader = data_provider(
            args = self.args,
            root_path=self.args.data_path,
            datasets=self.args.dataset,
            batch_size=self.args.batch_size,
            flag=flag,
            win_size = self.args.scale_win_size,
            step = self.args.scale_step,
            discrete=discrete,
            continuous_cols = self.continuous_cols,
            discrete_cols = self.discrete_cols,
        )
        if flag=='train' and discrete:
            self.continuous_cols = data_set.continuous_cols
            self.discrete_cols = data_set.discrete_cols
        return data_set, data_loader

    def _get_data_diffusion_ts(self, flag, discrete=True,):
        data_set, data_loader = data_provider(
            args = self.args,
            root_path=self.args.data_path,
            datasets=self.args.dataset,
            batch_size=self.args.batch_size,
            flag=flag,
            win_size = 100,
            step = 100,
            discrete=discrete,
            continuous_cols = self.continuous_cols,
            discrete_cols = self.discrete_cols,
        )
        if flag=='train' and discrete:
            self.continuous_cols = data_set.continuous_cols
            self.discrete_cols = data_set.discrete_cols
        return data_set, data_loader
    
    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

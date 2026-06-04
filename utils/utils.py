import numpy as np
import sys
import torch
from datetime import datetime
import torch.nn as nn
# import loralib as lora

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

from torch.utils.data import TensorDataset, DataLoader
class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device
        self.finish = False

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.finish = True
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            1: 1e-4, 3: 5e-5, 5:1e-5, 7: 5e-6, 9: 1e-6
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def name_with_datetime(prefix="default", with_time=True):
    if not with_time:
        return prefix
    now = datetime.now()
    return prefix + "-" + now.strftime("%Y%m%d")


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, verbose=False, delta=0):
        """

        Args:
            patience (int, optional): how many epochs to wait before stopping when loss is
               not improving. Defaults to 7.
            verbose (bool, optional): _description_. Defaults to False.
            delta (int, optional): minimum difference between new loss and old loss for
               new loss to be considered as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + f"checkpoint.pth")
        self.val_loss_min = val_loss

# class EarlyStopping:
#     """
#     Early stopping to stop the training when the loss does not improve after
#     certain epochs.
#     """
#
#     def __init__(self, patience=3, verbose=False, delta=0, lora=False):
#         """
#
#         Args:
#             patience (int, optional): how many epochs to wait before stopping when loss is
#                not improving. Defaults to 7.
#             verbose (bool, optional): _description_. Defaults to False.
#             delta (int, optional): minimum difference between new loss and old loss for
#                new loss to be considered as an improvement. Defaults to 0.
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.lora = lora
#         self.loss_score = None
#         self.f1_score = None
#         self.val_f1_max = -np.Inf
#
#     def __call__(self, val_loss, val_f1, model, path):
#         loss_score = -val_loss
#         f1_score = val_f1
#         score = loss_score + f1_score
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, val_f1, model, path)
#
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(
#                 f"EarlyStopping counter: {self.counter} out of {self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, val_f1, model, path)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, val_f1, model, path):
#         if self.verbose:
#             print(
#                 f"Validation loss decreased ({self.val_loss_min:.5f} --> {val_loss:.5f}).  "
#                 f"Validation f1 increased ({self.val_f1_max:.5f} --> {val_f1:.5f}).  Saving model ..."
#             )
#         if self.lora:
#             torch.save(lora.lora_state_dict(model), path + "/" + f"lora_checkpoint.pth")
#         else:
#             torch.save(model.state_dict(), path + "/" + f"checkpoint.pth")
#         self.val_loss_min = val_loss
#         self.val_f1_max = val_f1


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

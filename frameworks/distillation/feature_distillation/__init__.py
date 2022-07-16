from abc import abstractmethod

import torch

from model.layerwise_model import pad_const_channel


def get_distill_module(name):
    methods = {
        'L2Distillation': L2Distillation,
        'L1Distillation': L1Distillation,
        'FD_Conv1x1': FD_Conv1x1,
        'FD_Conv1x1_MSE': FD_Conv1x1_MSE,
        'KD': KD,
    }
    return methods[name]


class DistillationMethod(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, feat_s, feat_t, epoch_ratio):
        pass


class BridgeDistill(DistillationMethod):
    def __init__(self, bridge, distill_loss='MSE', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = bridge
        if distill_loss == 'MSE':
            self.loss = torch.nn.MSELoss()
        elif distill_loss == 'L1':
            self.loss = torch.nn.L1Loss()
        else:
            raise NotImplementedError()

    def forward(self, feat_s, feat_t, epoch_ratio):
        assert len(feat_s) == len(self.bridge)
        loss = []
        for fs, ft, b in zip(feat_s, feat_t, self.bridge):
            loss.append(self.loss(b(pad_const_channel(fs)), ft))
        return torch.mean(torch.stack(loss))


class KD(DistillationMethod):
    def __init__(self, *args, T=3, **kwargs):
        super().__init__()
        self.T = T

    def forward(self, feat_s, feat_t, epoch_ratio):
        import torch.nn.functional as F
        loss = 0
        cnt = 0
        for fs, ft in zip(feat_s, feat_t):
            if len(fs.shape) == 2 and fs.shape == ft.shape:
                p_s = F.log_softmax(fs / self.T, dim=1)
                p_t = F.softmax(ft / self.T, dim=1)
                loss += F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / fs.size(0)
                cnt += 1
        return loss / cnt


class L2Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean((fs - ft) ** 2))
        return torch.mean(torch.stack(loss))


class L1Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean(torch.abs(fs - ft)))
        return torch.mean(torch.stack(loss))


class FD_Conv1x1(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t) if
            len(fs.shape) == 4
        ])

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            loss.append(torch.mean(torch.abs(conv(fs) - ft)))
        return torch.mean(torch.stack(loss))


class FD_Conv1x1_MSE(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
            if len(fs.shape) == 4 and fs.shape[2:] == ft.shape[2:]
        ])

    def forward(self, feat_s, feat_t, epoch_ratio):
        loss = []
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            if len(fs.shape) == 4 and fs.shape[2:] == ft.shape[2:]:
                loss.append(torch.mean((conv(fs) - ft) ** 2))
        return torch.mean(torch.stack(loss))

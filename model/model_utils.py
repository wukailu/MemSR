import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn as nn, nn
from torch.utils.tensorboard.summary import hparams
from datasets import query_dataset


class Partial_Detach(nn.Module):
    def __init__(self, alpha=0):
        """
        When alpha = 0, it's completely detach, when alpha = 1, it's identity
        :param alpha:
        """
        super().__init__()
        self.alpha = 0

    def forward(self, inputs: torch.Tensor):
        if self.alpha == 0:
            return inputs.detach()
        elif self.alpha == 1:
            return inputs
        else:
            return inputs * self.alpha + inputs.detach() * (1 - self.alpha)


class Flatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=self.start_dim)


def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def std_alignment(model, x, fs_std=None):
    if fs_std is None:
        fs_std = []
    from model.layerwise_model import pad_const_channel
    from model.layerwise_model import ConvLayer
    assert isinstance(model, nn.ModuleList)
    assert isinstance(x, torch.Tensor)
    with torch.no_grad():
        for idx, m in enumerate(model):
            last_std = x.std()
            x = m(pad_const_channel(x))
            new_std = x.std()
            if isinstance(m, ConvLayer):
                if len(fs_std) != 0 and idx < len(fs_std):
                    print('last std = ', last_std, 'replaced by ', fs_std[idx])
                    last_std = fs_std[idx]
                else:
                    print('use last std as target std')
                m.conv.weight.data *= last_std/new_std
                x *= last_std/new_std


def SR_conv_init(conv: torch.nn.Conv2d):
    torch.nn.init.xavier_normal_(conv.weight.data, gain=1)
    conv.weight.data *= 2 ** 0.5
    return conv


def scale_weight(conv: torch.nn.Conv2d, ratio):
    conv.weight.data *= ratio
    return conv


def unfreeze_BN(model: torch.nn.Module):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = True


def freeze_BN(model: torch.nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = False


def get_trainable_params(model):
    # print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print("\t", repr(name))
            params_to_update.append(param)
    return params_to_update


def model_init(model: torch.nn.Module, method="kaiming_normal", activation='relu'):
    from torch import nn
    assert method in ['kaiming_normal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal']
    for name, ch in model.named_children():
        print(f"{name} is initialized using ", method)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            if method == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity=activation)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation))
            elif method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)) and m.weight.requires_grad:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MyTensorBoardLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
        # some alternative should be added
        self.tags.update(params)


def get_classifier(classifier, dataset: str) -> torch.nn.Module:
    if isinstance(dataset, str):
        dataset_type = query_dataset(dataset)
    elif isinstance(dataset, dict):
        dataset_type = query_dataset(dataset['name'])
    else:
        raise TypeError("dataset must be either str or dict")
    num_classes = dataset_type.num_classes
    if isinstance(classifier, str):
        classifier_name = classifier
        params = {}
    elif isinstance(classifier, dict):
        classifier_name = classifier['arch']
        params = {key: value for key, value in classifier.items() if key != 'arch'}
    else:
        raise TypeError('Classifier should be either str or a dict with at least a key "arch".')

    classifier_name = classifier_name.lower()
    if classifier_name.endswith("_sr"):
        from model.super_resolution_model import model_dict
        return model_dict[classifier_name[:-3]](num_classes=num_classes, **params)
    else:
        raise KeyError()


def load_models(hparams: dict) -> nn.ModuleList:
    num = len(hparams["pretrain_paths"])
    models: nn.ModuleList = nn.ModuleList([])
    for idx in range(num):
        checkpoint = torch.load(hparams["pretrain_paths"][idx], map_location='cpu')
        try:
            # If it's a lightning model
            last_param = checkpoint['hyper_parameters']
            if 'dataset' in hparams and hparams['dataset'] != 'concat':
                if last_param.dataset != hparams['dataset']:
                    print(
                        f"WARNING!!!!!!! Model trained on {last_param.dataset} will run on {hparams['dataset']}!!!!!!!")
                assert query_dataset(last_param.dataset).num_classes == query_dataset(
                    hparams['dataset']).num_classes

            model = get_classifier(last_param.backbone, last_param.dataset)
            model.load_state_dict({key[6:]: value for key, value in checkpoint["state_dict"].items()})
        except RuntimeError as e:
            print("RuntimeError when loading models", e)
            model = get_classifier(hparams["classifiers"][idx], hparams["dataset"])
            model.load_state_dict(checkpoint["model"])
        except TypeError as e:
            print("TypeError when loading models", e)
            # Maybe it's just a torch.save(model) and torch.load(model)
            model = checkpoint
        models.append(model)
    return models


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params:,}')
    return pytorch_total_params


def default_conv(in_channels, out_channels, kernel_size, bias=True, stride=1):
    if isinstance(kernel_size, int):
        pad_x, pad_y = kernel_size // 2, kernel_size // 2
    elif isinstance(kernel_size, tuple):
        pad_x, pad_y = kernel_size[0] // 2, kernel_size[1] // 2
    else:
        raise NotImplementedError()
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(pad_x, pad_y), bias=bias, stride=stride)


def convbn_to_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    bn.eval()
    out_channel, in_channel, kernel_size, _ = conv.weight.shape

    var = bn.running_var.data
    weight = bn.weight.data
    gamma = weight / torch.sqrt(var + bn.eps)

    bias = 0 if conv.bias is None else conv.bias.data

    conv_data = conv.weight.data * gamma.reshape((-1, 1, 1, 1))
    bias = bn.bias.data + (bias - bn.running_mean.data) * gamma

    ret = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True,
                    padding_mode=conv.padding_mode)
    ret.weight.data = conv_data
    ret.bias.data = bias
    return ret


def matmul_on_first_two_dim(m1: torch.Tensor, m2: torch.Tensor):
    """
    take matmul on first two dim only, and regard other dim as a scalar
    :param m1: tensor with at least 2 dim
    :param m2: tensor with at least 2 dim
    """
    if len(m1.shape) == 2:
        assert len(m2.shape) >= 2
        shape = m2.shape
        m2 = m2.flatten(start_dim=2).permute((2, 0, 1))
        ret = (m1 @ m2).permute((1, 2, 0))
        return ret.reshape(list(ret.shape[:2]) + list(shape[2:]))
    elif len(m2.shape) == 2:
        return matmul_on_first_two_dim(m2.transpose(0, 1), m1.transpose(0, 1)).transpose(0, 1)
    else:
        raise NotImplementedError()


def init_conv_with_conv(conv_t, conv_s, M):
    assert isinstance(conv_s, torch.nn.Conv2d)
    assert isinstance(conv_t, torch.nn.Conv2d)
    assert conv_s.stride == conv_t.stride
    assert conv_s.kernel_size == conv_t.kernel_size
    # 忽略Bias 误差 1e-5~1e-6, Bias = M^-1 Bias 误差 1e-2
    # 把 Kernel 看成一个 element 是向量的矩阵就行

    t_kernel = matmul_on_first_two_dim(conv_t.weight.data, M)

    r = conv_s.out_channels
    if conv_t.out_channels != r:
        u, s, v = torch.svd(t_kernel.flatten(start_dim=1))  # u and v are real orthogonal matrices
        M = u[:, :r]
        s_kernel = (torch.diag(s[:r]) @ v.T[:r]).reshape(conv_s.weight.shape)
    else:  # try to return ID when teacher and student has the same output channel
        s_kernel = t_kernel
        M = torch.eye(r)

    conv_s.weight.data = s_kernel
    if conv_t.bias is None:
        if conv_s.bias is not None:
            conv_s.bias.data = 0
    elif conv_s.bias is not None and conv_s.bias is not None:  # 如果老师有 bias, 学生也有 bias
        s_bias = M.pinverse() @ conv_t.bias.data
        conv_s.bias.data = s_bias
    else:
        raise AttributeError("conv_s do not have bias while conv_t has bias, which is not possible to init s with t")
    return M

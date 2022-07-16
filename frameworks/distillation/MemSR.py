# Distill Everything Into a Plain model

import torch
from torch import nn
from abc import ABC, abstractmethod

from frameworks.distillation.feature_distillation import get_distill_module
from frameworks.lightning_base_model import LightningModule, _Module
from model import freeze, std_alignment
from model.layerwise_model import ConvertibleLayer, ConvertibleModel, pad_const_channel, ConvLayer, IdLayer, \
    merge_1x1_and_3x3


class MemSR_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.M_maps = []
        self.teacher_plain_model, self.teacher = self.load_teacher()
        freeze(self.teacher_plain_model.eval())

        self.plain_model = nn.ModuleList()
        self.fs_std = []
        self.example_data = torch.stack(
            [self.unpack_batch(self.dataProvider.train_dl.dataset[i])[0] for i in range(16)], dim=0)

        import time
        start_time = time.process_time()
        self.init_student()
        print("initialization student width used ", time.process_time() - start_time)
        if self.params['std_align']:
            std_alignment(self.plain_model, self.example_data, self.fs_std)
            print("std alignment finished")

    def load_teacher(self):
        teacher = _Module.load_from_checkpoint(checkpoint_path=self.params['teacher_pretrain_path']).model
        assert isinstance(teacher, ConvertibleModel)
        return ConvertibleModel(teacher.to_convertible_layers()), teacher

    @abstractmethod
    def init_student(self):
        pass

    def init_layer(self, layer_s, layer_t, M):  # M is of shape C_t x C_s
        assert isinstance(layer_t, ConvertibleLayer)
        if 'normal' in self.params['layer_type']:
            M = layer_t.init_student(layer_s, M)
            return M
        else:
            raise NotImplementedError()

    def complete_hparams(self):
        default_sr_list = {
            'task': 'super-resolution',
            'input_channel': 3,
            'progressive_distillation': False,
            'init_with_teacher_param': False,
            'rank_eps': 5e-2,
            'layer_type': 'normal',
            'init_stu_with_teacher': False,
            'teacher_pretrain_path': None,
            'init_tail': False,
            'std_align': False,
            'fix_r': -1,
            'conv_init': "kaiming_normal",
        }
        self.params = {**default_sr_list, **self.params}
        LightningModule.complete_hparams(self)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.plain_model[start_forward_from: until]:
            x = m(pad_const_channel(x))
            if with_feature:
                f_list.append(x)
        return (f_list, x) if with_feature else x

    def unpack_batch(self, batch):
        if self.params['task'] == 'super-resolution':
            images, labels, filenames = batch
        else:
            raise NotImplementedError()
        return images, labels

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss

    def append_layer(self, in_channels, out_channels, previous_f_size, current_f_size, kernel_size=3):
        assert len(previous_f_size) == 2
        assert len(current_f_size) == 2
        if self.params['layer_type'].startswith('normal'):
            if 'prelu' in self.params['layer_type']:
                act = nn.PReLU()
            else:
                act = nn.ReLU()
            bn = 'no_bn' not in self.params['layer_type']
            if previous_f_size == current_f_size:
                stride = 1
            else:
                stride_w = previous_f_size[0] // current_f_size[0]
                stride_h = previous_f_size[1] // current_f_size[1]
                assert stride_h >= 1
                assert stride_w >= 1
                stride = (stride_w, stride_h)

            new_layer = ConvLayer(in_channels, out_channels, kernel_size, bn=bn, act=act, stride=stride,
                                  SR_init=self.params['task'] == 'super-resolution')
        else:
            raise NotImplementedError()

        self.plain_model.append(new_layer)

    def append_tail(self, last_channel, output_channel):
        if self.params['task'] == 'super-resolution':
            from model.super_resolution_model.edsr_layerwise_model import EDSRTail, EDSREasyTail
            if isinstance(self.teacher_plain_model.sequential_models[-1], EDSRTail):
                self.plain_model.append(
                    EDSRTail(self.params['scale'], last_channel, output_channel, 3, self.params['rgb_range']))
            elif isinstance(self.teacher_plain_model.sequential_models[-1], EDSREasyTail):
                self.plain_model.append(
                    EDSREasyTail(self.params['scale'], last_channel, output_channel, 3, self.params['rgb_range']))
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


def test_rank(r, M, f2, f, with_solution, with_bias, with_rank, bias, eps, adjust, ret_err=False):
    ret = []
    app_M = M[:, :r]
    app_f = f2[:r]

    approx = torch.mm(app_M, app_f)
    error = torch.norm(f - approx, p=2) / torch.norm(f, p=2)

    if with_bias and adjust != 0:
        # adjust the app_f to most positive value
        neg = app_f.clone()
        neg[neg > 0] = 0
        adj = -neg.mean(dim=1, keepdim=True) * adjust
        app_f = app_f + adj
        bias -= app_M @ adj

    if error < eps:
        ret.append(True)
    else:
        ret.append(False)
    if with_solution:
        ret.append(app_M)
        ret.append(app_f)
    if with_bias:
        ret.append(bias)
    if with_rank:
        ret.append(r)
    if not ret_err:
        return ret
    else:
        return ret, error


def rank_estimate(f, eps=5e-2, with_rank=True, with_bias=False, with_solution=False, fix_r=-1, adjust=3):
    """
    Estimate the size of feature map to approximate this. The return matrix f' should be positive if possible
    :param adjust: 0, does not adjust fs, otherwise adjust it with adjust * mean
    :param fix_r: just fix the returned width as r = fix_r to get the decomposition results
    :param use_NMF: whether use NMF instead of SVD
    :param with_rank: shall we return rank of f'
    :param with_solution: shall we return f = M f'
    :param with_bias: whether normalize f to zero-mean
    :param f: tensor of shape (C, N)
    :param eps: the error bar for low_rank approximation
    """
    #  Here Simple SVD is used, which is the best approximation to min_{D'} ||D-D'||_F where rank(D') <= r
    #  A question is, how to solve min_{D'} ||(D-D')*W||_F where rank(D') <= r,
    #  W is matrix with positive weights and * is element-wise production
    #  refer to wiki, it's called `Weighted low-rank approximation problems`, which does not have an analytic solution
    assert len(f.shape) == 2
    # svd can not solve too large feature maps, so take samples
    if f.size(1) > 32768:
        perm = torch.randperm(f.size(1))[:32768]
        f = f[:, perm]

    if with_bias:
        bias = f.mean(dim=1, keepdim=True)
        f -= bias
    else:
        bias = torch.zeros_like(f).mean(dim=1, keepdim=True)

    u, s, v = torch.svd(f)
    M = u
    f2 = torch.mm(torch.diag(s), v.t())

    if fix_r != -1:
        R = min(fix_r, f.size(0))
    else:
        L, R = 0, f.size(0)
        step = 1
        while L + step < R:
            ret = test_rank(L + step, M, f2, f, with_solution, with_bias, with_rank, bias, eps, adjust)
            if ret[0]:
                R = L + step
                break
            else:
                step *= 2
        L = step // 2
        step = step // 2
        while step != 0:
            if L + step < R:
                ret = test_rank(L + step, M, f2, f, with_solution, with_bias, with_rank, bias, eps, adjust)
                if not ret[0]:
                    L = L + step
                else:
                    R = L + step
            step = step // 2

    final_ret, error = test_rank(R, M, f2, f, with_solution, with_bias, with_rank, bias, eps, adjust,
                                 ret_err=True)
    print(f"Approximation error is {error}")

    if len(final_ret) == 2:
        return final_ret[1]
    else:
        return final_ret[1:]


class MemSR_Distillation(MemSR_LightModel, ABC):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.dist_method = self.get_distillation_module()
        if self.params['fix_distill_module']:
            freeze(self.dist_method)

    def get_distillation_module(self):
        sample, _ = self.unpack_batch(self.train_dataloader().dataset[0])
        sample = sample.unsqueeze(dim=0)
        with torch.no_grad():
            feat_t, out_t = self.teacher(sample, with_feature=True)
            feat_s, out_s = self(sample, with_feature=True)
            distill_config = self.params['dist_method']
            dist_method = get_distill_module(distill_config['name'])(feat_s, feat_t, **distill_config)
        return dist_method

    def complete_hparams(self):
        default_sr_list = {
            'dist_method': 'FD_Conv1x1_MSE',
            'fix_distill_module': False,
            'distill_coe': 0,
            'distill_alpha': 1e-5,
            'distill_coe_mod': 'old',
        }
        self.params = {**default_sr_list, **self.params}
        MemSR_LightModel.complete_hparams(self)
        if isinstance(self.params['dist_method'], str):
            self.params['dist_method'] = {'name': self.params['dist_method']}

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        if self.training:
            feat_s, predictions = self(images, with_feature=True)
            task_loss = self.criterion(predictions, labels)
            self.log('train/task_loss', task_loss, sync_dist=True)

            if self.params['distill_coe'] != 0:
                with torch.no_grad():
                    feat_t, out_t = self.teacher(images, with_feature=True)
                assert len(feat_s) == len(feat_t)
                ratio = self.current_epoch / self.params['num_epochs']
                dist_loss = self.dist_method(feat_s, feat_t, ratio)

                coe_task = 1
                coe_dist = self.params['distill_coe'] * (self.params['distill_alpha'] ** ratio)
                if self.params['distill_coe_mod'] != 'old':
                    coe_sum = (coe_task + coe_dist)
                    coe_task /= coe_sum
                    coe_dist /= coe_sum

                loss = task_loss * coe_task + dist_loss * coe_dist
                self.log('train/dist_loss', dist_loss, sync_dist=True)
                self.log('train/coe_task', coe_task, sync_dist=True)
                self.log('train/coe_dist', coe_dist, sync_dist=True)
            else:
                loss = task_loss
        else:
            predictions = self.forward(images)
            loss = self.criterion(predictions, labels)
            if self.params['distill_coe'] != 0 and self.teacher is not None:
                teacher_pred = self.teacher(images)
                metric = self.metric(teacher_pred, labels)
                self.log(phase + '/' + 'teacher_' + self.params['metric'], metric, sync_dist=True)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss


class MemSR_Init(MemSR_Distillation):
    def complete_hparams(self):
        default_sr_list = {
            'dist_method': 'BridgeDistill',
            'ridge_alpha': 0,
            'decompose_adjust': 3,
            'distill_init': True,
        }
        self.params = {**default_sr_list, **self.params}
        MemSR_Distillation.complete_hparams(self)

    def get_distillation_module(self):
        distill_config = self.params['dist_method']
        assert distill_config['name'] == 'BridgeDistill'
        from frameworks.distillation.feature_distillation import BridgeDistill
        return BridgeDistill(self.bridges[1:], **distill_config)

    def init_student(self):
        widths = [self.params['input_channel']]
        self.bridges = nn.ModuleList([IdLayer(self.params['input_channel'])])

        self.fs_his = []
        self.ft_his = []

        with torch.no_grad():
            f_list, _ = self.teacher(self.example_data, with_feature=True)
            self.ft_his = f_list
            teacher_width = [f.size(1) for f in f_list]
            for f in f_list[:-2]:
                mat = f.transpose(0, 1).flatten(start_dim=1)

                # M*fs + bias \approx mat
                M, fs, bias, r = rank_estimate(mat, eps=self.params['rank_eps'], with_bias=True, with_rank=True,
                                               with_solution=True, fix_r=self.params['fix_r'],
                                               adjust=self.params['decompose_adjust'])
                conv1x1 = nn.Conv2d(fs.size(0), mat.size(0), kernel_size=1, bias=True)
                conv1x1.weight.data[:] = M.reshape_as(conv1x1.weight)
                conv1x1.bias.data[:] = bias.reshape_as(conv1x1.bias)

                self.fs_std.append(fs.std())
                self.fs_his.append(fs)

                # print('---------layer ', len(self.bridges), '--------')
                # print('mat_shape', mat.shape, 'mat_min', mat.min(), 'mat_mean', mat.mean(), 'mat_std', f.std())
                # print('ft_shape', f.shape, 'f_min', f.min(), 'f_mean', f.mean(), 'f_std', f.std())
                # print('fs_shape', fs.shape, 'fs_min', fs.min(), 'fs_mean', fs.mean(), 'fs_std', fs.std())
                # print('M_shape', M.shape, 'M_min', M.min(), 'M_mean', M.mean(), 'M_std', M.std())
                # print('bias_shape', bias.shape, 'bias_min', bias.min(), 'bias_mean', bias.mean(), 'bias_std',
                #       bias.std())

                self.bridges.append(ConvLayer.fromConv2D(conv1x1))
                widths.append(r)
            self.fs_std.append(f_list[-2].std())
        widths.append(teacher_width[-2])
        self.bridges.append(IdLayer(teacher_width[-2]))
        widths.append(teacher_width[-1])
        self.bridges.append(IdLayer(teacher_width[-1]))
        print("calculated teacher width = ", [self.params['input_channel']] + teacher_width)
        print("calculated student width = ", widths)

        if not self.params['distill_init']:
            # reset the parameters in distill to random
            new_bridge = nn.ModuleList()
            for m in self.bridges:
                if isinstance(m, IdLayer):
                    new_bridge.append(ConvLayer(m.channel, m.channel, 1))
                elif isinstance(m, ConvLayer):
                    new_bridge.append(ConvLayer(m.conv.in_channels - 1, m.conv.out_channels, 1))
                else:
                    raise NotImplementedError()
            self.bridges = new_bridge

        f_shapes = [self.example_data.shape[-2:]] + [f.shape[-2:] for f in f_list[:-1]]
        with torch.no_grad():
            for i in range(len(self.bridges) - 2):
                if self.params['init_stu_with_teacher']:
                    print('Initializing layer...')
                    eq_conv, act = merge_1x1_and_3x3(self.bridges[i], self.teacher_plain_model[i]).simplify_layer()
                    conv = nn.Conv2d(widths[i] + 1, widths[i + 1], kernel_size=eq_conv.kernel_size,
                                     stride=eq_conv.stride,
                                     padding=eq_conv.padding, bias=False)
                    B = eq_conv.weight.data
                    M = self.bridges[i + 1].simplify_layer()[0].weight.data.flatten(start_dim=1)
                    M, bias = M[:, 1:], M[:, 0]
                    B[:, 0, B.size(2) // 2, B.size(3) // 2] -= bias
                    B = B.flatten(start_dim=1)
                    # solve MX=B
                    X = torch.lstsq(B, M)[0][:M.size(1)]
                    # print('B_mean', B.mean(), 'B_std', B.std(), 'B_max', B.max(), 'B_min', B.min())
                    # print('M_mean', M.mean(), 'M_std', M.std(), 'M_max', M.max(), 'M_min', M.min())
                    # print('X_mean', X.mean(), 'X_std', X.std(), 'X_max', X.max(), 'X_min', X.min())

                    conv.weight.data[:] = X.reshape_as(conv.weight)
                    self.plain_model.append(
                        ConvLayer.fromConv2D(conv, act=act, const_channel_0=True, version=self.params['layer_type']))
                else:
                    self.append_layer(widths[i], widths[i + 1], f_shapes[i], f_shapes[i + 1])
        self.append_tail(widths[-2], widths[-1])
        if self.params['init_stu_with_teacher'] or self.params['init_tail']:
            self.teacher_plain_model.sequential_models[-1].init_student(self.plain_model[-1], torch.eye(widths[-2]))
            print('Tail initialized')


def load_model(params):
    methods = {
        'DirectTrain': MemSR_LightModel,
        'Distillation': MemSR_Distillation,
        'MemSR_Init': MemSR_Init,
    }

    if 'load_from' in params:
        method = torch.load(params['load_from'], map_location=torch.device('cpu'))['hyper_parameters']['method']
        return methods[method].load_from_checkpoint(params['load_from'], strict=False)

    params = {'method': 'DirectTrain', **params}
    print("using method ", params['method'])
    model = methods[params['method']](params)
    return model

from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl
import torchmetrics

import meter.super_resolution_meter
from datasets.dataProvider import DataProvider
from model import get_classifier
from copy import deepcopy
from meter.super_resolution_meter import PSNR_SHAVE, SSIM_SHAVE

__all__ = ["LightningModule", "_Module"]


class LightningModule(pl.LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()  # must name after hparams or there will be plenty of bugs in early version of lightning
        self.save_hyperparameters(hparams)
        self.params = deepcopy(hparams)
        self.complete_hparams()
        self.criterion = self.choose_loss()
        self.dataProvider = DataProvider(params=deepcopy(self.params['dataset']))
        self.steps_per_epoch = len(self.train_dataloader().dataset) // self.params['dataset']["total_batch_size"]
        self.metric = self.choose_metric()
        self.need_to_learn = True
        self.train_results = {}
        self.val_results = {}
        self.test_results = {}

    def choose_metric(self):
        metric_map = {
            'acc': torchmetrics.Accuracy(),
            'psnr': torchmetrics.image.PSNR(),
            'psnr255': torchmetrics.image.PSNR(data_range=255),
            'psnr_shave_x4': PSNR_SHAVE(scale=4, gray=False, data_range=255),
            'psnr_gray_shave_x4': PSNR_SHAVE(scale=4, gray=True, data_range=255),
            'psnr_shave_x3': PSNR_SHAVE(scale=3, gray=False, data_range=255),
            'psnr_gray_shave_x3': PSNR_SHAVE(scale=3, gray=True, data_range=255),
            'psnr_shave_x2': PSNR_SHAVE(scale=2, gray=False, data_range=255),
            'psnr_gray_shave_x2': PSNR_SHAVE(scale=2, gray=True, data_range=255),
            'ssim_shave_x4': SSIM_SHAVE(scale=4, gray=False, data_range=255),
            'ssim_gray_shave_x4': SSIM_SHAVE(scale=4, gray=True, data_range=255),
            'ssim_shave_x3': SSIM_SHAVE(scale=3, gray=False, data_range=255),
            'ssim_gray_shave_x3': SSIM_SHAVE(scale=3, gray=True, data_range=255),
            'ssim_shave_x2': SSIM_SHAVE(scale=2, gray=False, data_range=255),
            'ssim_gray_shave_x2': SSIM_SHAVE(scale=2, gray=True, data_range=255),
        }
        return metric_map[self.params['metric'].lower()]

    def get_parameters_generator(self):
        return self.parameters

    def complete_hparams(self):
        default_list = {
            'optimizer': 'SGD',
            'lr_scheduler': 'OneCycLR',
            'max_lr': 0.1,
            'weight_decay': 5e-4,
            'step_decay': 0.1,
            'loss': 'CrossEntropy',
            'metric': 'acc',
        }
        default_dataset_values = {
            'batch_size': 128,
            'total_batch_size': 128,
        }
        self.params = {**default_list, **self.params}
        self.params['dataset'] = {**default_dataset_values, **self.params['dataset']}

    def choose_loss(self):
        loss_dict = {
            'CrossEntropy': torch.nn.CrossEntropyLoss,
            'L1': torch.nn.L1Loss,
            'MSE': torch.nn.MSELoss,
        }
        if self.params['loss'] in loss_dict:
            return loss_dict[self.params['loss']]()
        return None

    def choose_optimizer(self):
        gen = self.get_parameters_generator()
        if len(list(gen())) == 0:
            self.need_to_learn = False
            return None

        params = gen()
        from torch.optim import SGD, Adam
        if self.params['optimizer'] == 'SGD':
            optimizer = SGD(params, lr=self.params["max_lr"],
                            weight_decay=self.params["weight_decay"],
                            momentum=0.9, nesterov=True)
        elif self.params['optimizer'] == 'Adam':
            optimizer = Adam(params, lr=self.params['max_lr'],
                             weight_decay=self.params['weight_decay'])
        else:
            assert False, "optimizer not implemented"
        return optimizer

    def choose_scheduler(self, optimizer):
        if optimizer is None:
            return None

        from torch.optim import lr_scheduler
        if self.params['lr_scheduler'] == 'ExpLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        elif self.params['lr_scheduler'] == 'CosLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20 * self.steps_per_epoch + 1, eta_min=0)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.params['lr_scheduler'] == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        elif self.params['lr_scheduler'] == 'StepLR100':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        elif self.params['lr_scheduler'] == 'OneCycLR':
            # + 1 to avoid over flow in steps() when there's totally 800 steps specified and 801 steps called
            # there will be such errors.
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.params["max_lr"],
                                                steps_per_epoch=self.steps_per_epoch + 1,
                                                epochs=self.params["num_epochs"])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.params['lr_scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[70, 140, 190], gamma=0.1)
        elif self.params['lr_scheduler'] == 'MultiStepLR_EDSR_300':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
        else:
            return None
        return scheduler

    def configure_optimizers(self):
        optimizer = self.choose_optimizer()
        scheduler = self.choose_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    def train_dataloader(self):
        from copy import deepcopy
        return deepcopy(self.dataProvider.train_dl)

    def val_dataloader(self):
        from copy import deepcopy
        return deepcopy(self.dataProvider.val_dl)

    def test_dataloader(self):
        from copy import deepcopy
        return deepcopy(self.dataProvider.test_dl)

    @abstractmethod
    def forward(self, x):
        return x

    def step(self, batch, phase: str):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'validation')

    def test_step(self, batch, batch_nb):
        return self.step(batch, 'test')

    def on_test_end(self):
        import copy
        return copy.deepcopy(self.trainer.logger_connector.callback_metrics)


class _Module(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)  # must name after hparams or there will be plenty of bugs
        self.model = get_classifier(hparams["backbone"], hparams["dataset"])

    def forward(self, images):
        return self.model(images)

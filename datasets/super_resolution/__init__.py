from importlib import import_module
from torch.utils.data import dataloader
from datasets.utils import FullDatasetBase


class Data:
    def __init__(self, name, data_dir, data_train='', data_test='', test_only=False, **kwargs):
        if data_train == '' and data_test == '':
            data_train = data_test = name

        if data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            from datasets.super_resolution.benchmark import Benchmark
            testset = Benchmark(data_dir, name=name, test_only=test_only, train=False, **kwargs)
        else:
            if data_train.startswith('DIV2K-Q'):
                from datasets.super_resolution.div2kjpeg import DIV2KJPEGDataset
                testset = DIV2KJPEGDataset(data_dir, name=name, train=False, test_only=test_only, **kwargs)
            elif data_train == 'DIV2K':
                from datasets.super_resolution.div2kdataset import DIV2KDataset
                testset = DIV2KDataset(data_dir, name=name, train=False, test_only=test_only, **kwargs)
            else:
                raise NotImplementedError("This dataset is not implemented!")
        self.test_dataset = testset

        if test_only:
            self.train_dataset = self.test_dataset
        else:
            if data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
                from datasets.super_resolution.benchmark import Benchmark
                dataset = Benchmark(data_dir, name=name, test_only=test_only, train=False, **kwargs)
            elif data_train.startswith('DIV2K-Q'):
                from datasets.super_resolution.div2kjpeg import DIV2KJPEGDataset
                dataset = DIV2KJPEGDataset(data_dir, name=name, test_only=test_only, **kwargs)
            elif data_train == 'DIV2K':
                from datasets.super_resolution.div2kdataset import DIV2KDataset
                dataset = DIV2KDataset(data_dir, name=name, test_only=test_only, **kwargs)
            else:
                raise NotImplementedError("This dataset is not implemented!")

            self.train_dataset = dataset


# for SR dataset, there is no normalize initially, so data range is 0-255
# also, horizontal flip and vertical flip and transpose is applied with half probability
class SRFullDataset(FullDatasetBase):
    mean = (0, 0, 0)
    std = (1, 1, 1)
    img_shape = (3, 192, 192)
    num_classes = 1
    name = "SuperResolution"

    def __init__(self, name, data_dir='/data', **kwargs):
        super().__init__(**kwargs)
        self.ds = Data(name, data_dir, **kwargs)

    def gen_train_datasets(self, transform=None, target_transform=None):
        return self.ds.train_dataset

    def gen_val_datasets(self, transform=None, target_transform=None):
        return self.ds.test_dataset

    def gen_test_datasets(self, transform=None, target_transform=None):
        return self.ds.test_dataset

    @staticmethod
    def is_dataset_name(name: str):
        import re
        return re.match("(div2k-q[\d]*|div2k|set5|set14|b100|urban100)$", name.lower())

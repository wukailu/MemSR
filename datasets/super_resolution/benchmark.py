import os
from datasets.super_resolution import srdata


class Benchmark(srdata.SRData):
    def __init__(self, dir_data, train=True, **kwargs):
        super(Benchmark, self).__init__(dir_data, train=train, benchmark=True, **kwargs)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')


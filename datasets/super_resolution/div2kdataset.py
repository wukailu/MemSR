import os
from datasets.super_resolution.srdata import SRData


class DIV2KDataset(SRData):
    def __init__(self, dir_data, data_range='1-800/801-810', test_only=False, name='DIV2K', train=True, **kwargs):
        data_range = [r.split('-') for r in data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super().__init__(dir_data, name=name, train=train, **kwargs)

    def _scan(self):
        names_hr, names_lr = super(DIV2KDataset, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2KDataset, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large:
            self.dir_lr += 'L'


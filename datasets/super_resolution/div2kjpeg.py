import os
from datasets.super_resolution import div2kdataset


class DIV2KJPEGDataset(div2kdataset.DIV2KDataset):
    def __init__(self, dir_data, data_range='1-800/801-810', test_only=False, name='', train=True, **kwargs):
        self.q_factor = int(name.replace('DIV2K-Q', ''))
        super().__init__(dir_data, data_range=data_range, test_only=test_only, name='', train=train, **kwargs)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(
            self.apath, 'DIV2K_Q{}'.format(self.q_factor)
        )
        if self.input_large:
            self.dir_lr += 'L'
        self.ext = ('.png', '.jpg')

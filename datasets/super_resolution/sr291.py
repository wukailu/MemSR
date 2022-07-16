from datasets.super_resolution import srdata


class SR291(srdata.SRData):
    def __init__(self, dir_data, name='SR291', **kwargs):
        super(SR291, self).__init__(dir_data, name=name, **kwargs)

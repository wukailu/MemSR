import os

from datasets.super_resolution import common

from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT
import torch.utils.data as data


class Video(data.Dataset):
    def __init__(self, dir_demo, n_colors=3, rgb_range=255, scale=(4, ), name='Video', train=False, benchmark=False):
        self.rgb_range = rgb_range
        self.n_colors = n_colors
        self.name = name
        self.scale = scale
        self.idx_scale = 0
        self.train = False
        self.do_eval = False
        self.benchmark = benchmark

        self.filename, _ = os.path.splitext(os.path.basename(dir_demo))
        self.vidcap = VideoCapture(dir_demo)
        self.n_frames = 0
        self.total_frames = int(self.vidcap.get(CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        success, lr = self.vidcap.read()
        if success:
            self.n_frames += 1
            lr, = common.set_channel(lr, n_channels=self.n_colors)
            lr_t, = common.np2Tensor(lr, rgb_range=self.rgb_range)

            return lr_t, -1, '{}_{:0>5}'.format(self.filename, self.n_frames)
        else:
            self.vidcap.release()
            return None

    def __len__(self):
        return self.total_frames

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

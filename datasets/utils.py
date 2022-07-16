from torch.utils.data import Dataset
import torch
from torchvision import transforms
from abc import abstractmethod
from torch import nn, Tensor


class ExampleDataset(Dataset):
    def __init__(self, length=233):
        self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return index


class FullDatasetBase:
    mean: tuple
    std: tuple
    img_shape: tuple
    num_classes: int
    name: str

    def __init__(self, **kwargs):
        pass

    def gen_base_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]), None

    def gen_test_transforms(self):
        base, _ = self.gen_base_transforms()
        return base, _

    @abstractmethod
    def gen_train_transforms(self):
        return transforms.Compose([]), None

    @abstractmethod
    def gen_train_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_val_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    @abstractmethod
    def gen_test_datasets(self, transform=None, target_transform=None) -> Dataset:
        pass

    def sample_imgs(self) -> torch.Tensor:
        return torch.stack([torch.zeros(self.img_shape)] * 2)

    @staticmethod
    @abstractmethod
    def is_dataset_name(name: str):
        return name == "my_dataset"


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, input: Tensor):
        # x should be in shape of [N, C, H, W]
        assert input.dim() == 4
        return input * self.std + self.mean


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, input: Tensor):
        # x should be in shape of [N, C, H, W]
        assert input.dim() == 4
        # print(input.device, self.mean.device)
        return (input - self.mean) / self.std

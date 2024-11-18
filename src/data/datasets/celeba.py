from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
import os

class CelebA(Dataset):
    def __init__(self, data_root, split, transform = None):
        self.dataset = torchvision.datasets.CelebA(root=data_root, split=split, target_type="attr", download=False)
        self.list_attributes = [18, 31, 21]
        self.transform = transform
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        if self.transform is not None:
            input = self.transform(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

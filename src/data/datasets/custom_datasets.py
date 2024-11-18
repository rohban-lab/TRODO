from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np
from .neg_transformations import get_cutpaste, get_distort, get_elastic, get_mixup, get_rot

class NegativeDataset(Dataset):
    def __init__(self, base_dataset, label, neg_transformations=[], sequential=False, **kwargs):
        self.base_dataset = base_dataset
        self.label = label
        self.sequential = sequential
        self.transforms_order = neg_transformations
        self.transforms = {}
        for transform in neg_transformations:
            if transform == 'elastic':
                self.transforms['elastic'] = get_elastic(**kwargs.get('elastic', {}))
            elif transform == 'mixup':
                self.transforms['mixup'] = get_mixup(**kwargs.get('mixup', {}))
            elif transform == 'cutpaste':
                self.transforms['cutpaste'] = get_cutpaste(**kwargs.get('cutpaste', {}))
            elif transform == 'distort':
                self.transforms['distort'] = get_distort(**kwargs.get('distort', {}))
            elif transform == 'rot':
                self.transforms['rot'] = get_rot(**kwargs.get('rot', {}))
                

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        if self.sequential:
            def transform_func(image):
                for transform in self.transforms_order:
                    image = transform(image)
                    
                return image
            
            transform = transform_func
        else:
            transform = self.transforms[np.random.choice(list(self.transforms.keys()))]
        
        return transform(image), self.label


class MixedDataset(Dataset):
    def __init__(self, datasets, label, length, transform=None, datasets_probs=None):
        '''
        prob_dist is a probability distribution, according to which, a sample from datasets is selected
        '''
        self.datasets = datasets
        self.transform = transform
        self.length = length
        self.label = label
        
        self.datasets_probs = None
        if datasets_probs is not None:
            assert len(datasets_probs) == len(datasets)
            self.datasets_probs = datasets_probs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target_dataset = self.datasets[np.random.choice(len(self.datasets), p=self.datasets_probs)]
        sample_idx = np.random.randint(len(target_dataset))
        sample, _ = target_dataset[sample_idx]
        
        if self.transform is not None:
                
            to_pil = transforms.ToPILImage()
            sample = to_pil(sample)
    
            sample = self.transform(sample)
        
        return sample, self.label

class SingleLabelDataset(Dataset):
    def __init__(self, label, dataset):
        self.dataset = dataset
        self.len = len(dataset)
        self.label = label

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        return image, self.label

    def __len__(self):
        return self.len

class DummyDataset(Dataset):
    def __init__(self, label, pattern, pattern_args={}, transform=None):
        num_samples = pattern_args.get('num_samples', 1000)
        size = pattern_args.get('size', 32)
        channels = pattern_args.get('channels', 3)
        
        if pattern == 'gaussian':
            self.data = torch.randn((num_samples, channels, size, size))
        elif pattern == 'blank':
            color = pattern_args.get('color', 0)
            self.data = torch.ones((num_samples, channels, size, size)) * color
        elif pattern == 'uniform':
            low = pattern_args.get('low', 0)
            high = pattern_args.get('high', 1)
            self.data = torch.rand((num_samples, channels, size, size)) * (high - low) + low
        else:
            raise ValueError('Invalid pattern')
        self.label = label
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label
    

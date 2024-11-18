import torch
import torchvision
from torchvision import transforms
from .datasets.custom_datasets import MixedDataset, SingleLabelDataset, DummyDataset, NegativeDataset
from .datasets.gtsrb import GTSRB
from .datasets.pubfig import PubFig
from torchvision.datasets import ImageFolder
from .transforms import *
from .utils import sample_dataset
from ..constants import OUT_LABEL, IN_LABEL, TINY_IMAGENET_ROOT
import os
from torch.utils.data import DataLoader

ROOT = '~/data'
ALL_HARSH_TRANSFORMATIONS = ['rot', 'mixup', 'cutpaste', 'distort', 'elastic']

def get_transform(dataset):
  if dataset in ['cifar10', 'cifar100', 'gtsrb', 'SVHN']:
      return normal_transform
  elif dataset in ['stl10']:
      return no_transform
  elif dataset in ['fmnist', 'mnist']:
      return bw_transform
  elif dataset in ['gaussian', 'blank']:
      return None
  elif dataset in ['pubfig']:
      return hr_transform
  else:
      return None

def get_dataset(name, transform=None, train=True,
                dummy_params={}, download=False, in_dataset=None, **kwargs):
    '''
    Available datasets:
    - 'cifar10'
    - 'cifar100'
    - 'gtsrb'
    - 'mnist'
    - 'pubfig'
    - 'fmnist'
    - 'SVHN'
    - 'gaussian'
    - 'blank'
    - 'uniform'
    - 'stl'
    - 'TI'
    '''
    
    if transform is None:
        transform = get_transform(name)
        
        # Make sure ID samples and OOD samples have the same size
        if in_dataset is not None:
            id_sample = in_dataset[0][0]
            size = id_sample.size()[-1]
            channels = id_sample.size()[0]
            new_transforms = []
            if name != "TI":
                new_transforms.append(transforms.ToPILImage())
            
            if channels == 1:
                new_transforms.append(transforms.Grayscale())
            elif channels == 3 and name in ['mnist', 'fmnist']:
                new_transforms.append(transforms.Grayscale(3))
            new_transforms.append(transforms.Resize((size, size)))
            new_transforms.append(transforms.ToTensor())
            
            if transform is not None:
                transform = transforms.Compose([transform, transforms.Compose(new_transforms)])
            else:
                transform = transforms.Compose(new_transforms)
    try:
        if name == 'SVHN':
            return torchvision.datasets.SVHN(root=ROOT, split='train' if train else 'test', download=download, transform=transform)
        elif name == 'stl10':
            return torchvision.datasets.STL10(root=ROOT, split='train' if train else 'test', download=download, transform=transform)
        elif name == 'TI':
            return torchvision.datasets.ImageFolder(root=TINY_IMAGENET_ROOT, transform=transform)
        elif name == 'mnist':
            return torchvision.datasets.MNIST(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'fmnist':
            return torchvision.datasets.FashionMNIST(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'cifar10':
            return torchvision.datasets.CIFAR10(root=ROOT, train=train,transform=transform, download=download)
        elif name =='cifar100':
            return torchvision.datasets.CIFAR100(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'gtsrb':
            return GTSRB(train=train,transform=transform, download=download)
        elif name == 'pubfig':
            return PubFig(train=train, transform=transform)
        elif name in ['gaussian', 'blank', 'uniform']:
            label = dummy_params.get('label', OUT_LABEL)
            dummy_params['size'] = size
            dummy_params['channels'] = channels
            return DummyDataset(pattern=name, label=label, pattern_args=dummy_params)
        elif os.path.isdir(name):
            return ImageFolder(name, transform=hr_transform)
        else:
            raise NotImplementedError
    except Exception as e:
        if not download:
            return get_dataset(name, transform, train, dummy_params, download=True)
        else:
            raise e

def get_near_ood_loader(source_dataset=None,
                   batch_size=256,
                   **kwargs):
    in_dataset = get_dataset(source_dataset, trian=True, **kwargs)
    
    in_dataset = sample_dataset(in_dataset)

    harsh_transformations = ["rotation", "cutpaste"]

    # Out-Distribution Dataset

    out_dataset = NegativeDataset(base_dataset=in_dataset, label=OUT_LABEL,
                                neg_transformations=harsh_transformations, **kwargs)
    
    testloader = DataLoader(out_dataset,
                            num_workers=1,
                            batch_size=batch_size,
                            shuffle=True)
    
    # Sanity Check
    next(iter(testloader))
    
    return testloader

def get_cls_loader(dataset, train=False, sample_portion=0.2, batch_size=256, transforms_list=None):
    transform = None
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    if isinstance(dataset, str):
        test_dataset = get_dataset(dataset, transform, train)
    else:
        test_dataset = dataset
    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = DataLoader(test_dataset,
                            num_workers=1,
                            batch_size=batch_size,
                            shuffle=True)
    
    # Sanity Check
    next(iter(testloader))

    return testloader

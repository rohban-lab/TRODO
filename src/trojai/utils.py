import random
import torch
import pandas as pd
import warnings
import os
from copy import deepcopy

from torchvision import transforms
from torchvision.models import inception_v3
from ..models.base_model import BaseModel as Model
from torch.utils.data import DataLoader
from .dataset import ExampleDataset
from ..data.loaders import get_ood_loader
from ..data.utils import sample_dataset
from collections import defaultdict
from torch.utils.data import Subset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

archs_batch_sizes = {
    'squeezenetv1_1': 128,
    'squeezenetv1_0': 128,
    'shufflenet1_0': 128,
    'shufflenet1_5': 128,
    'shufflenet2_0': 128,
    # 'googlenet': 128,
    
    'default': 64,
    'resnet18': 128,
    
    'resnet50': 32,
    'resnet101': 32,
    'densenet121': 32,
    'inceptionv3': 64,
    'vgg19bn': 32,
    'vgg16bn': 32,
    'vgg13bn': 16,
    'vgg11bn': 32,
    'wideresnet101': 16,
    'wideresnet50': 32,
    'resnet152': 16,
    'densenet201': 16,
    'densenet161': 8,
    'densenet169':16,
}

def load_model(model_data, **model_kwargs):
    model_path = model_data['model_path']
    arch = model_data['arch']
    num_classes = model_data['num_classes']
    print("Loading a", arch)
    
    try:
        net = torch.load(model_path, map_location=device)
    except Exception as e:
        print("facing problems while loading this model", str(e))
        return None
    
    
    if arch == 'inceptionv3':
        new_net = inception_v3(num_classes=num_classes)
        new_net.load_state_dict(deepcopy(net.state_dict()), strict=False)
        net = new_net
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model


def get_dataset_trojai(model):
    rnd = model.meta_data['rnd']
    # if rnd < 3:
    #     example_data_path = 'example_data'
    # elif rnd == 4:
    #     example_data_path = 'clean_example_data'
    # else:
    #     example_data_path = 'clean-example-data'
    if rnd == 1:
        example_data_path = 'clean-example-data'
    else:
        example_data_path = 'example_data'
        
    return ExampleDataset(root_dir=os.path.join(os.path.dirname(model.meta_data['model_path']),
                          example_data_path), use_bgr=model.meta_data['bgr'], rnd=rnd)


def get_sanityloader_trojai(model, batch_size=None):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    return DataLoader(get_dataset_trojai(model), shuffle=True, batch_size=batch_size)


def get_oodloader_trojai(model, out_dataset, sample_num=None, batch_size=None, **kwargs):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    dataset = get_dataset_trojai(model)
    if sample_num:
        sample_num = min(sample_num, len(dataset))
        dataset = sample_dataset(dataset, portion=sample_num)
    
    return get_ood_loader(custom_in_dataset=dataset,
                          out_dataset=out_dataset,
                          out_transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                          batch_size=batch_size, **kwargs)

def split_dataset_by_arch(dataset):
    indices = defaultdict(lambda : [])
    all_archs = set()
    for i, model_data in enumerate(dataset.model_data):
        indices[model_data['arch']].append(i)
        all_archs.add(model_data['arch'])
    
    # Reorder the indices in each arch, so that the labels are alternating: 0, 1, 0 , 1, ...
    # To do this, first obtain the indices of the models with label 0 and 1
    # Then, reorder them by taking the first element of the first list, then the first element of the second list, then the second element of the first list, etc.
    for arch in all_archs:
        indices[arch] = sorted(indices[arch])
        label_0_indices = [i for i in indices[arch] if dataset.model_data[i]['label'] == 0]
        label_1_indices = [i for i in indices[arch] if dataset.model_data[i]['label'] == 1]
        new_indices = []
        for i in range(min(len(label_0_indices), len(label_1_indices))):
            new_indices.append(label_0_indices[i])
            new_indices.append(label_1_indices[i])
            
        if len(label_0_indices) > len(label_1_indices):
            new_indices += label_0_indices[len(label_1_indices):]
        else:
            new_indices += label_1_indices[len(label_0_indices):]
            
        indices[arch] = new_indices
    
    return {
        arch: Subset(dataset, arch_indices) for arch, arch_indices in indices.items()
    }

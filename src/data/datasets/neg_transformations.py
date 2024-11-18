from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, resize, to_grayscale, to_pil_image, to_tensor
import torchvision
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np
from .cutpaste import CutPasteUnion
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ElasticTransform
from torchvision.transforms import InterpolationMode

def get_random_param(param):
    if isinstance(param, tuple) or isinstance(param, list):
        if isinstance(param[0], int):
            return random.randint(*param)
        else:
            return random.uniform(*param)
    else:
        return param
    
def get_mixup(**kwargs):
    mixup_alpha = kwargs.get('mixup_alpha', 0.3)
    imagenet_root = kwargs.get('imagenet_root')
    if imagenet_root is None:
        raise ValueError('imagenet_root must be provided for mixup')
    
    imagenet_dataset = ImageFolder(imagenet_root,
                    transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
    def mixup(image, imagenet_dataset, mixup_alpha):
        imagenet_idx = np.random.randint(len(imagenet_dataset))
        imagenet_img, _ = imagenet_dataset[imagenet_idx]

        if imagenet_img.size()[-1] != image.size()[-1]:
            imagenet_img = resize(imagenet_img, (image.size()[-2], image.size()[-1]))
            
        if imagenet_img.size()[0] != image.size()[0]:
            imagenet_img = to_tensor(to_grayscale(to_pil_image(imagenet_img), image.size()[0]))
        new_mixup_alpha = get_random_param(mixup_alpha)

        mixed_img = (1 - new_mixup_alpha) * image + new_mixup_alpha * imagenet_img

        return mixed_img
    return lambda image: mixup(image, imagenet_dataset, mixup_alpha)

def get_elastic(**kwargs):
    p = kwargs.get('p', 1.0)
    alpha = kwargs.get('alpha', 150.0)
    sigma = kwargs.get('sigma', 5.0)
    alpha_affine = kwargs.get('alpha_affine', 10)
    interp = kwargs.get('interp', "bilinear") # or nearest
    
    # to_pil = transforms.ToPILImage()
    # elastic = transforms.Compose([ElasticTransform(alpha=alpha, sigma=sigma,
    #                            interpolation=InterpolationMode.BILINEAR if interp == 'bilinear' else InterpolationMode.NEAREST),
    #                               transforms.ToTensor()])
    # return lambda image: elastic(to_pil(image))
    def elastic(image):
        elastic_transform = A.Compose([A.ElasticTransform(alpha=get_random_param(alpha), p=get_random_param(p),
                                                sigma=get_random_param(sigma), alpha_affine=get_random_param(alpha_affine)),
                ToTensorV2()])
        return elastic_transform(image=image.permute(1, 2, 0).numpy())['image']
        
    return elastic
    

def get_cutpaste(**kwargs):
    cutpaste = CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ]), **kwargs)
    to_pil = transforms.ToPILImage()
    return lambda image: cutpaste(to_pil(image))

def get_distort(**kwargs):
    p = kwargs.get('p', 1.0)
    num_steps = kwargs.get('num_steps', 10)
    distort_limit = kwargs.get('distort_limit', 1)
    
    
    def distort(image):
        distort_transform = A.Compose([A.GridDistortion(num_steps=get_random_param(num_steps),
                                                        distort_limit=get_random_param(distort_limit),
                                                        p=get_random_param(p)),
            ToTensorV2()])
        return distort_transform(image=image.permute(1, 2, 0).numpy())['image']
    
    return distort

def get_rot(**kwargs):
    return lambda image: torch.rot90(image, k=1, dims=(1, 2))
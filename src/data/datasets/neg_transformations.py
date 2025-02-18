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
import math

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

def get_gridmask(**kwargs):
    d1 = kwargs.get('d1', 96)
    d2 = kwargs.get('d2', 224)
    rotate = kwargs.get('rotate', 1)
    ratio = kwargs.get('ratio', 0.6)
    
    def gridmask(image):
        # Handle tensor input
        if isinstance(image, torch.Tensor):
            is_tensor = True
            # Save original shape
            original_shape = image.shape
            # Convert to numpy, maintaining channel order
            image_np = image.cpu().numpy()
            
            # Handle different channel configurations
            if len(image_np.shape) == 3:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert to (H, W, C)
        else:
            is_tensor = False
            image_np = np.array(image)
        
        h, w = image_np.shape[:2]
        
        # Create mask
        mask = np.ones((h, w), np.float32)
        d = np.random.randint(d1, d2)
        l = int(d * ratio + 0.5)
        
        if rotate:
            angle = np.random.randint(0, 360)
            mask = Image.fromarray(np.uint8(mask * 255))
            mask = mask.rotate(angle)
            mask = np.array(mask) / 255.
        
        # Apply grid pattern
        for i in range(0, h+d, d):
            for j in range(0, w+d, d):
                mask[i:min(i+l, h), j:min(j+l, w)] = 0
        
        # Expand mask dimensions to match image
        if len(image_np.shape) == 3:
            mask = mask[:,:,np.newaxis]
        
        # Apply mask to image
        masked = image_np * mask
        
        if is_tensor:
            # Convert back to tensor with original shape
            if len(original_shape) == 3:
                masked = np.transpose(masked, (2, 0, 1))  # Convert back to (C, H, W)
            return torch.from_numpy(masked.astype(np.float32))
        else:
            return Image.fromarray(np.uint8(masked))
    
    return gridmask

def get_random_erasing(**kwargs):
    p = kwargs.get('p', 0.5)
    scale = kwargs.get('scale', (0.02, 0.33))
    ratio = kwargs.get('ratio', (0.3, 3.3))
    
    def random_erasing(image):
        if random.random() > p:
            return image
        
        # Handle tensor input
        if isinstance(image, torch.Tensor):
            is_tensor = True
            # Save original shape and type
            original_shape = image.shape
            original_dtype = image.dtype
            # Convert to numpy, maintaining channel order
            image_np = image.cpu().numpy()
            
            # Handle different channel configurations
            if len(image_np.shape) == 3:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert to (H, W, C)
        else:
            is_tensor = False
            image_np = np.array(image)
        
        h, w = image_np.shape[:2]
        
        # Random rectangle parameters
        area = h * w
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])
        
        # Calculate dimensions
        h_rect = int(round(math.sqrt(target_area * aspect_ratio)))
        w_rect = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if h_rect < h and w_rect < w:
            x1 = random.randint(0, w - w_rect)
            y1 = random.randint(0, h - h_rect)
            
            # Random noise matching the number of channels
            if len(image_np.shape) == 3:
                channels = image_np.shape[2]
                noise_shape = (h_rect, w_rect, channels)
            else:
                noise_shape = (h_rect, w_rect)
            
            image_np[y1:y1+h_rect, x1:x1+w_rect] = np.random.rand(*noise_shape)
        
        if is_tensor:
            # Convert back to tensor with original shape
            if len(original_shape) == 3:
                image_np = np.transpose(image_np, (2, 0, 1))  # Convert back to (C, H, W)
            return torch.from_numpy(image_np.astype(np.float32))
        else:
            return Image.fromarray(np.uint8(image_np * 255))
    
    return random_erasing

def get_colorjitter_plus(**kwargs):
    brightness = kwargs.get('brightness', 0.4)
    contrast = kwargs.get('contrast', 0.4)
    saturation = kwargs.get('saturation', 0.4)
    hue = kwargs.get('hue', 0.1)
    p = kwargs.get('p', 0.8)
    
    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )
    
    def colorjitter_plus(image):
        if random.random() > p:
            return image
        
        if isinstance(image, torch.Tensor):
            # Apply color jitter directly to tensor
            return color_jitter(image)
        else:
            # Apply to PIL Image
            return color_jitter(image)
    
    return colorjitter_plus

def get_jigsaw(**kwargs):
    grid_size = kwargs.get('grid_size', 3)
    
    def jigsaw(image):
        # Handle tensor input
        if isinstance(image, torch.Tensor):
            is_tensor = True
            # Save original shape
            original_shape = image.shape
            # Convert to numpy, maintaining channel order
            image_np = image.cpu().numpy()
            
            # Handle different channel configurations
            if len(image_np.shape) == 3:  # (C, H, W)
                image_np = np.transpose(image_np, (1, 2, 0))  # Convert to (H, W, C)
        else:
            is_tensor = False
            image_np = np.array(image)
        
        h, w = image_np.shape[:2]
        patch_size_h = h // grid_size
        patch_size_w = w // grid_size
        
        # Split image into patches
        patches = []
        for i in range(grid_size):
            for j in range(grid_size):
                patch = image_np[i*patch_size_h:(i+1)*patch_size_h,
                               j*patch_size_w:(j+1)*patch_size_w]
                patches.append(patch)
        
        # Shuffle patches
        random.shuffle(patches)
        
        # Reconstruct image
        new_image = np.zeros_like(image_np)
        for i, patch in enumerate(patches):
            row = i // grid_size
            col = i % grid_size
            new_image[row*patch_size_h:(row+1)*patch_size_h,
                     col*patch_size_w:(col+1)*patch_size_w] = patch
        
        if is_tensor:
            # Convert back to tensor with original shape
            if len(original_shape) == 3:
                new_image = np.transpose(new_image, (2, 0, 1))  # Convert back to (C, H, W)
            return torch.from_numpy(new_image.astype(np.float32))
        else:
            return Image.fromarray(np.uint8(new_image))
    
    return jigsaw
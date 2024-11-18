import copy
import pickle 
from torchvision import transforms
import torch
from .base_model import BaseModel as Model
from .preact import PreActResNet18
from torchvision.models import resnet18
from torchvision.models import vit_b_16


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_preact(record_path, num_classes=10, **model_kwargs):
    
    load_file = torch.load(record_path)

    new_dict = copy.deepcopy(load_file['model'])
    for k, v in load_file['model'].items():
        if k.startswith('module.'):
            del new_dict[k]
            new_dict[k[7:]] = v

    load_file['model'] = new_dict
    
    net = PreActResNet18(num_classes=num_classes)
    net.load_state_dict(load_file['model'])
    
    model = Model(net, feature_extractor=net.get_features, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model

def load_resnet(record_path, num_classes=10, **model_kwargs):
    state_dict = torch.load(record_path)
    
    
    net = resnet18(num_classes=num_classes)
    net.load_state_dict(state_dict['model'])
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model


def load_vit(record_path, num_classes=10, **model_kwargs):
    state_dict = torch.load(record_path)
    new_dict = copy.deepcopy(state_dict['model'])
    for k, v in state_dict['model'].items():
        if k.startswith('module.'):
            del new_dict[k]
            new_dict[k[7:]] = v
    state_dict['model'] = new_dict
    good_dict = {}
    for x, y in new_dict.items():
        if x.startswith("1."):
            good_dict[x[2:]] = y
    vit_model = vit_b_16(num_classes=num_classes)
    vit_model.load_state_dict(good_dict)
    
    feature_extractor = torch.nn.Sequential(*list(vit_model.children())[:-1])

    model = Model(vit_model, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    return model

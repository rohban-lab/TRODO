import os, random
from torch.utils.data import Dataset
from copy import copy

CLEAN_LABEL = 0

class ModelDataset(Dataset):
    def __init__(self,
                 clean_folder,
                 trojaned_folder,
                 model_loader,
                 sample=False,
                 sample_portion=0.2,
                 ):
        
        self.model_loader = model_loader
        
        self.models_data = []
        for model_file in os.listdir(clean_folder):
            if model_file.lower().endswith(".pt"):
                self.models_data.append({
                    'path': os.path.join(clean_folder, model_file),
                    'label': CLEAN_LABEL,
                })
                
            
        for model_file in os.listdir(trojaned_folder):
            if model_file.lower().endswith(".pt"):
                self.models_data.append({
                    'path': os.path.join(trojaned_folder, model_file),
                    'label': 1 - CLEAN_LABEL,
                })
        
        random.shuffle(self.models_data)
        
        if sample:
            self.models_data = random.choices(self.models_data, k=int(len(self.models_data)* sample_portion))

    def __len__(self):
        return len(self.models_data)

    def __getitem__(self, idx):
        model_data = self.models_data[idx]
        model = self.model_loader(model_data['path'], model_data)
        label = model_data['label']
        
        return model, label
    
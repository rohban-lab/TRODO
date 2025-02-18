import random
import torch
import pandas as pd
import warnings
import os

from torch.utils.data import Dataset

warnings.simplefilter("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TrojAIDataset(Dataset):
    def __init__(self, root_dirs, rnd, model_loader, shuffle=True, data_csv=None,
                 size=224, use_bgr=False, sample=False, sample_portion=0.2, custom_arch=None, discard_arch=None,
                load_check=False):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.bgr = rnd in [0, 1]
        self.round = rnd
        self.model_loader = model_loader
        
        self.model_data = []
        
        for root_dir in root_dirs:
            try:
                names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
                
                data = pd.read_csv(data_csv)
                data.set_index('model_name', inplace=True)

                data = data.to_dict(orient='index')
                for name in names:
                    if (custom_arch and data[name]['model_architecture'] not in custom_arch) or (discard_arch and data[name]['model_architecture'] in discard_arch):
                        continue
                    model_path = os.path.join(root_dir, name, 'model.pt')
                    with open(os.path.join(root_dir, name, 'ground_truth.csv'), 'r') as f:
                        label = f.read()
                    self.model_data.append({
                        'rnd': rnd,
                        'name': name,
                        'bgr': self.bgr,
                        # 1 means trojaned and 0 means clean
                        'label': int(label),
                        'model_path': model_path,
                        'num_classes': data[name]['number_classes'],
                        'arch': data[name]['model_architecture'],
                        'data': data,
                    })
                    if load_check:
                        model = model_loader(self.model_data[-1])
                        if model is None:
                            self.model_data = self.model_data[:-1]
                        else:
                            del model
            except Exception as e:
                print(f"Error while loading models of directory: {root_dir} Error:", e)
                print("Skipping this directory")
        
        random.shuffle(self.model_data)
        if sample:
            self.model_data = random.sample(self.model_data, int(sample_portion * len(self.model_data)))
        
    def __len__(self):
        return len(self.model_data)

    def __getitem__(self, idx):
        try:
            model = self.model_loader(self.model_data[idx], meta_data=self.model_data[idx], normalize=False)
            return model, model.meta_data['label'] if model is not None else None
        except Exception as e:
            print("Following error occured while loading model", str(e))
            return None, None
    

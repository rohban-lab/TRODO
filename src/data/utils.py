import random
from torch.utils.data import Subset
from collections import defaultdict

def sample_dataset(dataset, portion=0.01, balanced=False):
    if portion>1:
        portion = portion / len(dataset)
    if not balanced:
        indices = random.sample(range(len(dataset)), int(portion * len(dataset)))
    # It is assumed that the dataset has labels
    else:
        indices = []
        labels = [y for _, y in dataset]
        unique_labels = list(set(labels))
        labels_indices = defaultdict(lambda : [])
        
        for i, label in enumerate(labels):
            labels_indices[label].append(i)
            
        for label in unique_labels:
            indices += random.sample(labels_indices[label], int(portion * len(labels_indices[label])))
        
    return Subset(dataset, indices)


def filter_labels(dataset, labels):
    indices = [i for i, (_, y) in enumerate(dataset) if y not in labels]
    return Subset(dataset, indices)
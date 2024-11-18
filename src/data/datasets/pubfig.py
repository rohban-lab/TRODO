from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class PubFig(Dataset):
    def __init__(self, root_dir='/kaggle/input/pubfig-50class-dataset/pubfig', train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train (bool): If True, load images from the train folder, else from the test folder.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.subfolder = 'train' if train else 'test'
        self.image_paths = []
        self.labels = []
        self.load_images()

    def load_images(self):
        """ Load all images from the specified subfolder in each category directory. """
        categories = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        for category_label, category_name in enumerate(categories):
            category_path = os.path.join(self.root_dir, category_name, self.subfolder)
            for img_filename in sorted(os.listdir(category_path)):
                if img_filename.lower().endswith('.jpg'):
                    self.image_paths.append(os.path.join(category_path, img_filename))
                    self.labels.append(category_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

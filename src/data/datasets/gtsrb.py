import  os
from PIL import Image
import csv
import torch.utils.data as data
import subprocess, os, stat

DATA_ROOT = '/data/gtsrb'

def run_download_bash_file(script_path):
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)
    subprocess.run(['bash', script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
class GTSRB(data.Dataset):
    def __init__(self, train, data_root=DATA_ROOT, transform=None, download=False):
        super(GTSRB, self).__init__()
        if not os.path.exists(data_root):
            run_download_bash_file(os.path.join(os.path.dirname(__file__),'gtsrb_download.sh'))
        if train:
            self.data_folder = os.path.join(data_root, "Train")
            self.images, self.labels = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(data_root, "Test")
            self.images, self.labels = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)

        self.transform = transform

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + '' + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label
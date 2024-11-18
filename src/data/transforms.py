import torchvision
from torchvision import transforms

normal_transform = transforms.Compose(
    [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])

celeba_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

no_transform = transforms.Compose([
    transforms.ToTensor()
    ])

bw_transform = transforms.Compose([
    transforms.ToTensor()
    ])
hr_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

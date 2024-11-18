# For OOD Data: 1 is for ID, 0 is for OOD
OUT_LABEL = 0
IN_LABEL = 1

TINY_IMAGENET_ROOT = "~/data/tinyimagenet/"

# Number of classes
num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'pubfig': 50,
    'fmnist': 10,
    'gtsrb': 43,
    'celeba': 8,
}

# Normalizations
NORM_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4865, 0.4409),
    'mnist': (0.5, 0.5, 0.5),
    'pubfig': (0.485, 0.456, 0.406),
    'gtsrb': (0, 0, 0),
    'celeba': (0, 0, 0),
}

NORM_STD = {
    'cifar10': (0.247, 0.243, 0.261),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'mnist': (0.5, 0.5, 0.5),
    'pubfig': (0.229, 0.224, 0.225),
    'gtsrb': (1, 1, 1),
    'celeba': (1, 1, 1),
}


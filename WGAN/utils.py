import os

import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


loader = T.Compose([T.ToTensor(),
                    T.Lambda(lambda t: (t * 2) - 1)])

unloader = T.Lambda(lambda t: (t + 1) / 2)


def get_loaders(config):
    train_data = MNIST(root="data/", train=True, download=True, transform=loader)
    test_data = MNIST(root="data/", train=False, download=True, transform=loader)

    train_loader = DataLoader(train_data, batch_size=config["bs"], num_workers=config["j"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=config["bs"], num_workers=config["j"], drop_last=True)

    return train_loader, test_loader


def make_default_dirs():
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
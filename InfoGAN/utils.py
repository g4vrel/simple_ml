import os

import torch

import torch.utils
import torchvision.transforms as T
from torchvision.datasets import MNIST

loader = T.Compose([
    T.ToTensor(),
])

def get_loaders(config):
    train_data = MNIST(root="data/", train=True, download=True, transform=loader)
    test_data = MNIST(root="data/", train=False, download=True, transform=loader)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["bs"], num_workers=config["j"], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["bs"], num_workers=config["j"], drop_last=True)

    return train_loader, test_loader

def make_default_dirs():
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
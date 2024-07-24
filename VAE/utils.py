import torch

import torch.utils
import torchvision.transforms as T
from torchvision.datasets import MNIST

loader = T.Compose([
    T.ToTensor(), # x \in [0, 1]
])

def get_loaders(config):
    transform = loader

    train_data = MNIST(root="data/", train=True, download=True, transform=transform)
    test_data = MNIST(root="data/", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["bs"], num_workers=config["j"], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["bs"], num_workers=config["j"], shuffle=True, drop_last=True) # test

    return train_loader, test_loader

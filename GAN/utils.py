import torch

import torch.utils
import torchvision.transforms as T
from torchvision.datasets import MNIST

loader = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda t: (t * 2) - 1)
])

unloader = T.Lambda(lambda t: (t + 1) / 2)

def get_loaders(config):
    transform = loader

    train_data, test_data = MNIST(root="data/", train=True, download=True, transform=transform), MNIST(root="data/", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["bs"], num_workers=config["j"], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["bs"], num_workers=config["j"], drop_last=True)

    return train_loader, test_loader
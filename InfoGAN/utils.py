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


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    checkpoint = {
        'epoch': int(epoch),
        'step': int(step),
        'model_state_dict': model.state_dict(),
    }

    if optim is not None:
        checkpoint['optim_state_dict'] = optim.state_dict()

    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    checkpoint = torch.load(path, weights_only=True)
    step = int(checkpoint['step'])
    epoch = int(checkpoint['epoch'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if optim is not None:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        ema_model.eval()

    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    model.eval()

    return step, epoch, model, optim, scaler, ema_model  
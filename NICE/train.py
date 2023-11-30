import os
import argparse
import numpy as np
from torchvision.utils import save_image, make_grid
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch
import torch.optim as optim
from modules import NICE, make_prior


os.makedirs('results', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)
parser = argparse.ArgumentParser(description='NICE')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=75, metavar='N',
                    help='number of epochs to train (default: 75)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='train on CUDA')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
print(f'=> Using {device} device')


def get_loaders():
    train = datasets.MNIST(root='data/', train=True, download=True, transform=T.ToTensor())
    test = datasets.MNIST(root='data/', train=False, download=True, transform=T.ToTensor())
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test, batch_size=args.test_batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    return train_loader, test_loader


def generate_sample(model, epoch):
    with torch.no_grad():
        _, x = model.sample(64)
        x = x.view(-1, 1, 28, 28)
        x = torch.clamp(x, 0, 1)
        x = make_grid(x)
        x = x.cpu()
        save_image(x, 'results/sample_' + str(epoch) + '.png')


def dequantization(x):
    x = 255. * x
    x = x + torch.rand_like(x, device=x.device)
    x = x / 256
    return x


def train(epoch, model, loader, optimizer):
    model.train()
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True).view(-1, 28*28)
        x = dequantization(x)
        z, prior_log_prob, log_det = model(x)
        log_prob = prior_log_prob + log_det
        loss = -torch.mean(log_prob) #NLL
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            _loss = loss.item() + 28*28*np.log(255)
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(loader.dataset),
                100. * batch_idx / len(loader), _loss))


def test(epoch, model, loader):
    model.eval()
    with torch.no_grad():
        k = 0
        mean_loss = 0.0
        for batch_idx, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True).view(-1, 28*28)
            x = dequantization(x)
            z, prior_log_prob, log_det = model(x)
            log_prob = prior_log_prob + log_det
            loss = -torch.mean(log_prob)
            mean_loss -= loss
            k += 1
        mean_loss /= k
        print('Epoch: {}\tTest set log likelihood: {:.6f}'.format(epoch, mean_loss))


if __name__ == '__main__':
    SAVE_MODEL = True
    run_config = argparse.Namespace(
        name='mnist_nice',
        lr=1e-3,
        gamma=0.9,
        nh=1000,
    )
    train_loader, test_loader = get_loaders()
    prior = make_prior(device=device)
    model = NICE(prior, nh=run_config.nh).to(device)
    optimizer = optim.Adam(model.parameters(), lr=run_config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=run_config.gamma)
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, optimizer)
        test(epoch, model, test_loader)
        scheduler.step()
        generate_sample(model, epoch)

    if SAVE_MODEL:
        path = f'saved_models/{run_config.name}.pt'
        torch.save(model.state_dict(), path)

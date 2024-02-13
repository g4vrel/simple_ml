from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import make_grid
import lightning as L
import torch


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str="data/", batch_size:int=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.ToTensor()
        self.j = 2

    def prepare_data(self):
        MNIST(self.data_dir, download=True, train=True, transform=self.transform)
        MNIST(self.data_dir, download=True, train=False, transform=self.transform)

    def setup(self, stage:str):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [50000, 10000],
                                                        generator=torch.Generator().manual_seed(159753)
                                                        )
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.j)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, drop_last=True, num_workers=self.j)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, drop_last=True, num_workers=self.j)
    

def output_images(x0: torch.Tensor, nrow=10):
    x0 = x0.view(-1, 1, 28, 28)
    x0 = torch.clamp(x0, 0, 1)
    x0 = make_grid(x0, nrow=nrow)
    im = x0.cpu()
    return T.ToPILImage()(im)
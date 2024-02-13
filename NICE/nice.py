import wandb
import torch
import torch.nn as nn
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import numpy as np
from modules import *


L.seed_everything(159753)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_wandb = True
if use_wandb:
    wandb.login()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'=> Using {device} device')
nin = 784
nh = 1000
nlayers = 5
lr = 2e-3
bs = 256


class MLP(nn.Module):
    def __init__(self, nin, nh):
        super().__init__()
        layers = [nn.Linear(nin, nh), nn.ReLU()]
        for _ in range(nlayers):
            layers += [nn.Linear(nh, nh), nn.ReLU()]
        layers += [nn.Linear(nh, nin)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ACL(nn.Module):
    def __init__(self, nin, nh, p):
        super().__init__()
        self.m = MLP(nin, nh)
        self.p = p
        self.nin = nin

    def forward(self, x):
        x1, x2 = x[:,::2], x[:,1::2]
        if self.p:
            x1, x2 = x2, x1
        z1 = x1
        z2 = x2 + self.m(x1)
        if self.p:
            z1, z2 = z2, z1
        z = torch.empty(x.shape, device=x.device)
        z[:, ::2] = z1
        z[:, 1::2] = z2
        return z

    def backward(self, z):
        z1, z2 =  z[:,::2], z[:,1::2]
        if self.p:
            z1, z2 = z2, z1
        x1 = z1
        x2 = z2 - self.m(z1)
        if self.p:
            x1, x2 = x2, x1
        x = torch.empty(z.shape, device=z.device)
        x[:, ::2] = x1
        x[:, 1::2] = x2
        return x
    

class NICE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.flow = nn.ModuleList([ACL(nin // 2, nh, (i%2 != 0)) for i in range(4)])
        self.prior = TransformedDistribution(Uniform(torch.zeros(nin, device=device), torch.ones(nin, device=device)),
                                             [SigmoidTransform().inv]
                                            )
        self.s = nn.Parameter(torch.rand(1, nin, requires_grad=True))
        self.dims = 28 * 28

    def forward(self, x, log_det=0):
        x = x.clone()
        for f in self.flow:
            x = f.forward(x)
        z = torch.exp(self.s) * x
        log_det += torch.sum(self.s)
        return z, log_det

    @torch.no_grad()
    def sample(self, shape:tuple=(100,)):
        self.eval()
        z = self.prior.sample(shape)
        x = z.clone() * torch.exp(-self.s)
        for f in self.flow[::-1]:
            x = f.backward(x)
        return x.view(-1, 28, 28)

    def _dequantize(self, x):
        return (255 * x + torch.rand_like(x).detach()) / 256, -x.shape[1] * np.log(256) * torch.ones(x.shape[0], device=x.device)

    def _process(self, x):
        x = x.view(-1, 28*28)
        x, log_det = self._dequantize(x)
        return x, log_det

    def _common_step(self, batch, batch_idx):
        x, log_det = self._process(batch[0])
        z, log_det = self(x, log_det)
        prior_log_prob = self.prior.log_prob(z).sum(dim=1)
        log_prob = log_det + prior_log_prob
        return log_prob

    def training_step(self, batch, batch_idx):
        log_prob = self._common_step(batch, batch_idx)
        loss = -torch.mean(log_prob)
        self.log('train_nll', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        log_prob = self._common_step(batch, batch_idx)
        loss = -torch.mean(log_prob)
        self.log('val_nll', loss)
        ll_fixed = torch.logsumexp(log_prob, dim=-1).mean() + self.dims * np.log(256)
        self.log('val_log_prob', ll_fixed, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        log_prob = self._common_step(batch, batch_idx)
        ll = torch.mean(log_prob)
        ll_fixed = torch.logsumexp(log_prob, dim=-1).mean() + self.dims * np.log(256)
        self.log('test_log_prob', ll_fixed, prog_bar=True, on_epoch=True)
        return ll

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    fast_dev_run = False
    train = True
    name = 'final'
    max_epochs = 300
    nice = NICE()
    mnist = MNISTDataModule(batch_size=bs)
    if use_wandb:
        logger = WandbLogger(name=name, project='NICE', log_model=False)
        logger.watch(nice, log_freq=500)
    else:
        logger = True
    trainer = L.Trainer(fast_dev_run=fast_dev_run,
                        max_epochs=max_epochs,
                        accelerator='gpu',
                        log_every_n_steps=50,
                        logger=logger,
                        callbacks=[LearningRateMonitor("epoch"),
                                   EarlyStopping(monitor='val_log_prob', mode='max', stopping_threshold=2000.0)])
    if train:
        trainer.fit(nice, datamodule=mnist)
        test_result = trainer.test(nice, datamodule=mnist)
    print(test_result)
    if use_wandb:
        wandb.finish()
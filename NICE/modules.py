import torch
import torch.nn as nn
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform


class MLP(nn.Module):
    def __init__(self, nin, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nin),
        )

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
        x2 = z2 - self.m(x1)
        if self.p:
            x1, x2 = x2, x1
        x = torch.empty(z.shape, device=z.device)
        x[:, ::2] = x1
        x[:, 1::2] = x2
        return x


class NICE(nn.Module):
    def __init__(self, nin=28*28, nh=1000, device='cuda'):
        super().__init__()
        self.nin = nin
        self.s = nn.Parameter(torch.rand(1, nin, requires_grad=True))
        self.flow = nn.ModuleList([
            ACL(nin//2, nh, False),
            ACL(nin//2, nh, True),
            ACL(nin//2, nh, False),
            ACL(nin//2, nh, True)
        ])
        base_distribution = Uniform(torch.zeros(nin, device=device), torch.ones(nin, device=device))
        transforms = [SigmoidTransform().inv]
        self.prior = TransformedDistribution(base_distribution, transforms)

    def forward(self, x):
        x = x.clone()
        for flow in self.flow:
            x = flow.forward(x)
        x = torch.exp(self.s) * x
        log_prob = self.prior.log_prob(x).sum(dim=1)
        log_det = torch.sum(self.s)
        return x, log_prob, log_det

    def backward(self, h):
        h = h.clone() * torch.exp(-self.s)
        for flow in self.flow[::-1]:
            h = flow.backward(h)
        return h, torch.sum(-self.s, dim=1)

    def sample(self, n=1):
        z = self.prior.sample((n,))
        x, _ = self.backward(z)
        return z, x
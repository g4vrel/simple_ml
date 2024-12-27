import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import OneHotCategorical, Normal


class Generator(nn.Module):
    def __init__(self, input_dim=74, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 28*28),
                                 nn.Sigmoid())

    def forward(self, noise):
        return self.net(noise)

class SharedNet(nn.Module):
    def __init__(self, cat_dim, cont_dim, hidden_dim=128):
        super().__init__()
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim

        self.net = nn.Sequential(nn.Linear(28*28, hidden_dim),
                                 nn.LeakyReLU(0.1),
                                 nn.Dropout(0.25),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.LeakyReLU(0.1))
        
        self.discriminator = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.recognition = nn.Sequential(nn.Linear(hidden_dim, 128),
                                         nn.BatchNorm1d(128),
                                         nn.LeakyReLU(0.1),
                                         nn.Linear(128, cat_dim + 2 * cont_dim),)

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)

    def discriminate(self, h):
        return self.discriminator(h)

    def recognize(self, h):
        h = self.recognition(h)
        
        shape = [self.cat_dim, self.cont_dim, self.cont_dim]
        logits = torch.split(h, shape, dim=-1)

        cat = F.softmax(logits[0], 1)
        mean = logits[1]
        logstd = logits[2]

        return {"cat": cat,
                "mean": mean,
                "logstd": logstd}


def sample_latent(config, device='cuda'):    
    cat = F.one_hot(torch.randint(0, 10, (config['bs'],), device=device), 10).float()
    cont = torch.empty((config['bs'], config['cont_dim']), device=device).uniform_(-1, 1)

    noise = torch.randn((config['bs'], config['noise_dim']), device=device)

    latent = {'cat': cat,
              'cont': cont,
              'noise': noise}

    return (torch.cat((cat, cont, noise), 1), latent)


def categorical_mi(latent, approximation, device='cuda'):
    probs = torch.ones(10, device=device) / 10
    
    prior = OneHotCategorical(probs)
    posterior = OneHotCategorical(probs=approximation['cat'])

    mutual_information = posterior.log_prob(latent['cat']) - prior.log_prob(latent['cat'])
    
    return mutual_information.mean()


def gaussian_mi(latent, approximation, device='cuda'):
    prior = Normal(torch.zeros_like(latent['cont']), torch.ones_like(latent['cont']))
    
    mean = approximation['mean']
    std = torch.exp(0.5 * approximation['logstd'])

    posterior = Normal(mean, std)

    mutual_information = posterior.log_prob(latent['cont']) - prior.log_prob(latent['cont'])
    
    return mutual_information.mean()


def lower_bound_loss(latent, approximation):
    categorical_loss = F.cross_entropy(approximation['cat'], latent['cat'])

    mean = approximation['mean']
    std = torch.exp(0.5 * approximation['logstd'])
    gaussian = torch.distributions.Normal(mean, std)

    continuous_loss = -gaussian.log_prob(latent['cont']).sum(dim=1).mean()

    return categorical_loss, continuous_loss
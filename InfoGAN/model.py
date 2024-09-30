import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=1, std=0.02)
        nn.init.constant_(m.bias, val=0)


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
        self.apply(init_weights)

    def forward(self, noise):
        return self.net(noise)


class SharedNet(nn.Module):
    def __init__(self, cat_dim, cont_dim, hidden_dim=128):
        super().__init__()
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim

        self.net = nn.Sequential(nn.Linear(28*28, hidden_dim),
                                 nn.LeakyReLU(negative_slope=0.1),
                                 nn.Dropout(0.25),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.LeakyReLU(negative_slope=0.1))
        
        self.discriminator = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.recognition = nn.Sequential(nn.Linear(hidden_dim, 128),
                                         nn.BatchNorm1d(128),
                                         nn.LeakyReLU(0.1),
                                         nn.Linear(128, cat_dim + 2 * cont_dim),)

        self.apply(init_weights)

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
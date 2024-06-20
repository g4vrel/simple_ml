import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1200, output_dim=28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),)

    def forward(self, noise):
        noise = noise.flatten(1)
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=240, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),)

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)
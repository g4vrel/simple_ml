import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, output_dim=28 * 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),)

    def forward(self, noise):
        return self.net(noise)


# X -> R
class Critic(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.flatten(1)
        return self.net(x)
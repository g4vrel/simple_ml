import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def gaussian_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        - 1.0
    )


def kl_objective(vae, x):
    pred, z, eps, mean, logvar = vae(x)

    likelihood_term = F.binary_cross_entropy(pred, x.flatten(1), reduction='none').sum(1).mean()
    kl_term = gaussian_kl(mean, logvar, torch.zeros_like(mean), torch.zeros_like(logvar)).sum(1).mean()

    assert likelihood_term.shape == kl_term.shape

    return likelihood_term, kl_term


class BernoulliVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.mu.weight)
        init.zeros_(self.mu.bias)
        init.xavier_uniform_(self.log_sigma.weight)
        init.constant_(self.log_sigma.bias, -5)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_sigma(h)

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = x.flatten(1)

        mean, logvar = self.encode(x)
        logstd = 0.5 * logvar
        
        eps = torch.randn_like(mean)
        z = mean + eps * logstd.exp()

        pred = self.decode(z)

        return pred, eps, z, mean, logvar
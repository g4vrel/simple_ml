import numpy as np

import torch

torch.manual_seed(159753)
np.random.seed(159753)

from torchvision.utils import save_image
import torchvision.transforms as T

from utils import get_loaders
from modules import BernoulliVAE, kl_objective

def make_model(config):
    if config["decoder"] == "bernoulli":
        vae = BernoulliVAE(
            input_dim=int(np.array(config["input_shape"]).prod()),
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"]
        )
    return vae

if __name__ == '__main__':
    import os

    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    device = 'cuda'
    
    config = {
        "decoder": "bernoulli",
        "objective": "kl",
        "lr":1e-3,
        "epochs": 50,
        "latent_dim": 16,
        "hidden_dim": 128,
        "input_shape": (28, 28),
        "bs":64,
        "j":3,
        "print_freq":100
    }

    vae = make_model(config).to(device)

    train_loader, eval_loader = get_loaders(config)

    optim = torch.optim.Adam(vae.parameters(), lr=config["lr"], betas=(0.5, 0.999))

    vae.train()

    step = 0

    for epoch in range(config['epochs']):
        for x, _ in train_loader:
            optim.zero_grad()

            x = x.to(device)

            likelihood, kl = kl_objective(vae, x)
            loss = likelihood + kl
            loss.backward()
            
            # TODO: figure out why norm is so large
            grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

            optim.step()
            
            if step % config["print_freq"] == 0:
                print("Step: {:4.0f} | Loss = {:.6f} | p_θ(x|z): {:.4f} | KL: {:.4f} | Grad_norm: {:.4f}".format(
                    step, loss.item(), likelihood.item(), kl.item(), grad_norm
                ))

            step += 1

        with torch.no_grad():
            vae.eval()

            loss = 0.0
            k = 0
            for x, _ in eval_loader:
                x = x.to(device)

                likelihood, kl = kl_objective(vae, x)
                loss += (likelihood + kl).mean().item()

                k += 1
            
            loss /= k
            print(f"Eval loss = {loss :.5f}")

            pred = vae(x)[0]

            n = min(x.size(0), 16)

            line = torch.zeros_like(x)[:n]

            comparison = torch.cat([x[:n], line, pred.view(config["bs"], 1, 28, 28)[:n]])
            save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png')

            noise = torch.randn(64, config["latent_dim"]).to(device)
            sample = vae.decode(noise)

            save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')

            vae.train()

    torch.save(vae.state_dict(), f'saved_models/vae_{step}.pt')
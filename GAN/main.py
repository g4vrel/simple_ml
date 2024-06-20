import numpy as np

import torch

torch.manual_seed(159753)
np.random.seed(159753)

from torchvision.utils import save_image
import torchvision.transforms as T

from utils import get_loaders, unloader
from modules import Generator, Discriminator


def make_models(config):
    generator = Generator(
        input_dim = config["noise_dim"],
        hidden_dim = config["gen_hidden"],
        output_dim = int(np.array(config["input_shape"]).prod())
    )

    discriminator = Discriminator(
        input_dim = int(np.array(config["input_shape"]).prod()),
        hidden_dim = config["dis_hidden"],
    )
    return generator, discriminator


def test_discriminator(config, generator, discriminator, eval_loader, device='cuda'):
    real_pred = 0.0
    fake_pred = 0.0

    for x, _ in eval_loader:
        x = x.to(device)
        z = torch.randn((config["bs"], config["noise_dim"]), device=x.device)

        real = discriminator(x) 
        fake = discriminator(generator(z))
        
        real_pred += torch.where(real > 0.5, 1.0, 0.0).sum()
        fake_pred += torch.where(fake < 0.5, 1.0, 0.0).sum()

    real_pred /= len(eval_loader.dataset)
    fake_pred /= len(eval_loader.dataset)

    print(f"Accuracy on real data: {real_pred.item():.4f}")
    print(f"Accuracy on fake data: {fake_pred.item():.4f}")


def gen_objective(config, discriminator, generator, z):
    if config["gen_loss"] == "non-saturating":
        return -discriminator(generator(z)).log()

    elif config["gen_loss"] == "zero-sum":
        return (1 - discriminator(generator(z))).log()


if __name__ == '__main__':
    import os

    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    device = 'cuda'
    
    config = {
        "gen_loss": "non-saturating",
        "lr_gen": 2e-4,
        "lr_dis": 2e-4,
        "epochs": 100,
        "noise_dim": 128,
        "gen_hidden": 512,
        "dis_hidden": 256,
        "input_shape": (28, 28),
        "bs":64,
        "j":3,
        "print_freq":100,
    }

    generator, discriminator = make_models(config)

    train_loader, eval_loader = get_loaders(config)

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=config["lr_gen"], betas=(0.5, 0.999))
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=config["lr_dis"], betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    step = 0

    for epoch in range(config['epochs']):
        for x, _ in train_loader:
            optim_dis.zero_grad()

            x = x.to(device)

            z = torch.randn((config["bs"], config["noise_dim"]), device=x.device)
            
            batch_loss = -discriminator(x).log() - (1 - discriminator(generator(z).detach())).log()
            dis_loss = batch_loss.mean()
            dis_loss.backward()
            
            dis_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

            optim_dis.step()
            
            optim_gen.zero_grad()
        
            z = torch.randn((config["bs"], config["noise_dim"]), device=x.device)

            batch_loss = gen_objective(config, discriminator, generator, z)
            gen_loss = batch_loss.mean()
            gen_loss.backward()

            gen_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

            optim_gen.step()

            if step % config["print_freq"] == 0:
                print(f"Epoch: {epoch:2.0f} | Dis_loss = {dis_loss.item() :.5f} | Gen_loss = {gen_loss.item():.5f} | Dis_norm: {dis_norm:.4f} | Gen_norm: {gen_norm:.4f}")

            step += 1

        with torch.no_grad():
            generator.eval()
            discriminator.eval()

            test_discriminator(config, generator, discriminator, eval_loader)

            sample = torch.randn(64, config["noise_dim"]).to(device)
            sample = generator(sample)
            
            sample = unloader(sample)

            save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')

            generator.train()
            discriminator.train()

    torch.save(generator.state_dict(), f'saved_models/generator_{step}.pt')
    torch.save(discriminator.state_dict(), f'saved_models/discriminator_{step}.pt')
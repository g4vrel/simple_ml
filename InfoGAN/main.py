import numpy as np

import torch
import torch.nn.functional as F

torch.manual_seed(159753)
np.random.seed(159753)

from model import Generator, SharedNet

from utils import get_loaders, make_default_dirs
from eval import img_grid

from torch.distributions import OneHotCategorical, Normal


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


if __name__ == '__main__':
    make_default_dirs()

    device = 'cuda'

    config = {
        'cat_dim': 10,
        'cont_dim': 2,
        'noise_dim': 62,
        'input_shape': (28, 28),
        'bs': 128,
        'j': 3,
        'print_freq': 200,
        'lambda_categorical': 1,
        'lambda_continuous': 0.1,
        'lr_dis': 2e-4,
        'lr_gen': 2e-4,
        'epochs': 200
    }

    generator = Generator(hidden_dim=512).to(device)

    shared_net = SharedNet(cat_dim=config['cat_dim'],
                           cont_dim=config['cont_dim']).to(device)

    train_loader, eval_loader = get_loaders(config)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=config['lr_gen'], betas=(0.5, 0.999))
    optim_sn = torch.optim.Adam(shared_net.parameters(), lr=config['lr_dis'], betas=(0.5, 0.999))

    adv_loss = torch.nn.BCELoss()

    ones = torch.ones((config['bs'], 1), device=device)
    zeros = torch.zeros((config['bs'], 1), device=device)

    step = 0
    for epoch in range(config['epochs']):
        generator.train()
        shared_net.train()

        for x, _ in train_loader:
            x = x.to(device)
            z, latent = sample_latent(config)

            gen_x = generator(z)

            # Discriminator step
            fake_repr = shared_net(gen_x.detach())
            real_repr = shared_net(x)

            dis_loss = (adv_loss(shared_net.discriminate(real_repr), ones) 
                        + adv_loss(shared_net.discriminate(fake_repr), zeros))

            optim_sn.zero_grad(set_to_none=True)
            dis_loss.backward()
            dis_norm = torch.nn.utils.clip_grad_norm_(shared_net.parameters(), 1.0)
            optim_sn.step()

            # Generator and mutual information step
            fake_repr = shared_net(gen_x)

            gen_loss = adv_loss(shared_net.discriminate(fake_repr), ones)

            approximation = shared_net.recognize(fake_repr)

            cat_mi = categorical_mi(latent, approximation)
            cont_mi = gaussian_mi(latent, approximation)

            information_loss = config['lambda_categorical'] * cat_mi + config['lambda_continuous'] * cont_mi

            total_loss = gen_loss - information_loss

            optim_gen.zero_grad(set_to_none=True)
            optim_sn.zero_grad(set_to_none=True)
            total_loss.backward()
            info_norm = torch.nn.utils.clip_grad_norm_(shared_net.parameters(), 1.0)
            optim_gen.step()
            optim_sn.step()
            
            if step % config['print_freq'] == 0:
                info_term = config['lambda_categorical'] * cat_mi + config['lambda_continuous'] * cont_mi
                print(f'{dis_loss.item():.5f} | {gen_loss.item():.5f} | {info_term.item():.5f}')

            step += 1

        with torch.no_grad():
            generator.eval()
            img_grid(str(epoch), generator, config)


    torch.save(generator.state_dict(), f'saved_models/generator_{step}.pt')
    torch.save(shared_net.state_dict(), f'saved_models/discriminator_{step}.pt')
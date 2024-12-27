import os

import torch
from model import *
from utils import get_loaders, make_checkpoint
from eval import img_grid


if __name__ == '__main__':
    os.makedirs('samples', exist_ok=True)

    device = 'cuda'

    config = {
        'cat_dim': 10,
        'cont_dim': 2,
        'noise_dim': 62,
        'input_shape': (28, 28),
        'bs': 128,
        'j': 6,
        'print_freq': 200,
        'lambda_categorical': 1,
        'lambda_continuous': 0.1,
        'lr_dis': 2e-4,
        'lr_gen': 2e-5,
        'epochs': 500
    }

    generator = Generator(hidden_dim=512).to(device)
    generator = torch.compile(generator)

    shared_net = SharedNet(hidden_dim=512,
                           cat_dim=config['cat_dim'],
                           cont_dim=config['cont_dim']).to(device)
    shared_net = torch.compile(shared_net)

    train_loader, eval_loader = get_loaders(config)

    optim_gen = torch.optim.Adam(generator.parameters(), lr=config['lr_gen'], betas=(0.5, 0.95))
    optim_sn = torch.optim.Adam(shared_net.parameters(), lr=config['lr_dis'], betas=(0.5, 0.95))

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
            dis_norm = torch.nn.utils.clip_grad_norm_(shared_net.parameters(), 0.1)
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
            info_norm = torch.nn.utils.clip_grad_norm_(shared_net.parameters(), 0.1)
            optim_gen.step()
            optim_sn.step()
            
            if step % config['print_freq'] == 0:
                info_term = config['lambda_categorical'] * cat_mi + config['lambda_continuous'] * cont_mi
                print(f'Step: {step} [{epoch}]: Dis_loss: {dis_loss.item():.5f} | Gen_loss: {gen_loss.item():.5f} | Info_term: {info_term.item():.5f}')

            step += 1

        with torch.no_grad():
            generator.eval()
            img_grid(str(epoch), generator, config)

    make_checkpoint(f'generator_{step}.tar',
                    step=step,
                    epoch=epoch,
                    model=generator,
                    optim=optim_gen)

    make_checkpoint(f'sharednet_{step}.tar',
                    step=step,
                    epoch=epoch,
                    model=shared_net,
                    optim=optim_sn)
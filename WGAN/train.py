import numpy as np

import torch

torch.manual_seed(159753)
np.random.seed(159753)

from model import Generator, Critic

from utils import get_loaders, make_default_dirs
from eval import generation


if __name__ == '__main__':
    make_default_dirs()

    device = 'cuda'

    config = {
        'noise_dim': 64,
        'input_shape': (28, 28),
        'bs': 64,
        'j': 3,
        'print_freq': 200,
        'lr': 0.0001,
        'epochs': 200,
        'c': 0.01,
        'n_critic': 5
    }

    generator = Generator().to(device)
    critic = Critic().to(device)

    train_loader, eval_loader = get_loaders(config)

    optim_gen = torch.optim.RMSprop(generator.parameters(), lr=config['lr'])
    optim_cri = torch.optim.RMSprop(critic.parameters(), lr=config['lr'])

    step = 0
    for epoch in range(config['epochs']):
        generator.train()
        critic.train()

        critic_losses = []
        for x, _ in train_loader:
            x = x.to(device)
            z = torch.randn((config['bs'], config['noise_dim']), device=device)

            gen_x = generator(z).detach()

            # Critic step
            critic_loss = - critic(x).mean() + critic(gen_x).mean()

            critic_losses.append(critic_loss)

            optim_cri.zero_grad(set_to_none=True)
            critic_loss.backward()
            optim_cri.step()

            with torch.no_grad():
                for p in critic.parameters():
                    p.data.clamp_(-config['c'], config['c'])

            # Generator step
            if (step + 1) % config['n_critic'] == 0:
                z = torch.randn((config['bs'], config['noise_dim']), device=device)
                gen_x = generator(z)

                generator_loss = - critic(gen_x).mean()

                optim_gen.zero_grad(set_to_none=True)
                generator_loss.backward()
                optim_gen.step()

            if (step + 1) % config['print_freq'] == 0:
                critic_mean_loss = torch.tensor(critic_losses).mean()
                print(f'{critic_mean_loss.item():.5f} | {generator_loss.item():.5f}')

                critic_losses = []

            step += 1

        with torch.no_grad():
            generator.eval()
            generation(config, str(epoch), generator)


    torch.save(generator.state_dict(), f'saved_models/generator_{step}.pt')
    torch.save(critic.state_dict(), f'saved_models/critic_{step}.pt')

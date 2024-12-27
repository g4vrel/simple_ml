import os
import torch

from model import Generator, Critic

from utils import get_loaders, unloader
from torchvision.utils import save_image


if __name__ == '__main__':
    os.makedirs('samples', exist_ok=True)

    device = 'cuda'

    config = {
        'noise_dim': 64,
        'input_shape': (28, 28),
        'bs': 64,
        'num_workers': 6,
        'print_freq': 400,
        'lr': 5e-5,
        'epochs': 1500,
        'c': 0.01,
        'n_critic': 5
    }

    generator = Generator().to(device)
    generator = torch.compile(generator)
    critic = Critic().to(device)
    critic = torch.compile(critic)

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
            
            shape = (64, config['noise_dim'])

            noise = torch.randn(shape, device=device)
            gen_x = generator(noise).view(64, 1, 28, 28)
            gen_x = unloader(gen_x)

            save_image(gen_x.cpu(), 'samples/sample_' + str(epoch) + '.png')

    torch.save(generator.state_dict(), f'generator_{step}.pt')
    torch.save(critic.state_dict(), f'critic_{step}.pt')

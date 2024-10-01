import torch
from torchvision.utils import save_image
from utils import unloader


@torch.no_grad()
def generation(config, epoch, generator, device='cuda'):    
    shape = (64, config['noise_dim'])

    noise = torch.randn(shape, device=device)
    gen_x = generator(noise).view(64, 1, 28, 28)
    gen_x = unloader(gen_x)

    save_image(gen_x.cpu(), 'results/generation_' + str(epoch) + '.png')
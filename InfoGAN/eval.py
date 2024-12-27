import numpy as np

import torch

torch.manual_seed(159753)
np.random.seed(159753)

from torchvision.utils import save_image
import torch.nn.functional as F


@torch.no_grad()
def img_grid(name, generator, config, device='cuda'):
    tensor = torch.zeros(100, device=device)

    for i in range(10):
        start_idx = i * 10
        end_idx = start_idx + 10
        tensor[start_idx:end_idx] = i

    cat = F.one_hot(tensor.long(), 10)
    c1 = torch.linspace(-1.0, 1.0, 10, device=device)
    c2 = torch.linspace(-1.0, 1.0, 10, device=device)
    c = torch.stack((c1, c2), dim=1)
    cont = c.repeat(10, 1)

    noise = torch.empty((1, config['noise_dim']), device=device).uniform_(-1, 1)
    noise = noise.repeat(100, 1)

    noise = torch.cat((cat, cont, noise), 1)

    gen_data = generator(noise)
    gen_data = gen_data.view(100, 1, 28, 28)

    save_image(gen_data.cpu(), 'results/grid_' + str(name) + '.png', nrow=10)
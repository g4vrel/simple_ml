import numpy as np

import torch

torch.manual_seed(159753)
np.random.seed(159753)

from torchvision.utils import save_image

from utils import unloader
from modules import Generator


if __name__ == '__main__':
    device = 'cuda'

    config = {
        "noise_dim": 128,
        "gen_hidden": 512,
        "input_shape": (28, 28),
    }

    generator = Generator(
        input_dim = config["noise_dim"],
        hidden_dim = config["gen_hidden"],
        output_dim = int(np.array(config["input_shape"]).prod())
    ).to(device)

    generator.load_state_dict(torch.load('saved_models/generator_93700.pt'))
    generator.eval()

    sample = torch.randn(64, config["noise_dim"]).to(device)
    sample = generator(sample)
    sample = unloader(sample)

    save_image(sample.view(64, 1, 28, 28), 'results/output.png')
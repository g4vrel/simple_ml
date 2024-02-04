import torch, torch.nn.functional as F
from einops import rearrange
from unet import Unet
from tqdm import tqdm


use_wandb = True
if use_wandb:
    import wandb


class DDPM:
    def __init__(self, timesteps=1000, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        self.denoiser = Unet().to(device)
        self._register_scheduler()
        self.ext = lambda x: rearrange(x, 'b -> b 1 1 1')
        self.step = 0
        self.epoch = 0

    def _register_scheduler(self):
        register_buffer = lambda name, val: setattr(self, name, val.to(torch.float32))
        betas = torch.linspace(0.0001, 0.02, self.timesteps, dtype=torch.float64, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = alphas.cumprod(0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        log_variance = torch.log(variance.clamp(min=1e-20))
        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        register_buffer('sqrt_m1_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('variance', variance)
        register_buffer('log_variance', log_variance)
        register_buffer('coef', betas / torch.sqrt(1.0 - alphas_cumprod))

    def checkpoint(self, opt, scheduler, loc):
        torch.save({
            'denoiser_state_dict': self.denoiser.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch
        }, loc)

    def load_checkpoint(self, checkpoint, optimizer, scheduler):
        checkpoint = torch.load(checkpoint)
        self.denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        self.step = int(checkpoint['step'])
        self.epoch = int(checkpoint['epoch'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def q_forward(self, x_0, noise, t):
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_m1_alphas_cumprod = self.sqrt_m1_alphas_cumprod[t]
        x_t = self.ext(sqrt_alphas_cumprod) * x_0 + self.ext(sqrt_m1_alphas_cumprod) * noise
        return x_t

    def denoise_step(self, x_t, pred_noise, t, noise):
        variance = torch.exp(0.5 * self.ext(self.log_variance[t]))
        sqrt_recip_alphas = self.ext(self.sqrt_recip_alphas[t])
        coef = self.ext(self.coef[t])
        x_prev = sqrt_recip_alphas * (x_t - coef * pred_noise) + variance * noise
        return x_prev

    def train(self, loader, optimizer, scheduler, epochs, log_freq=3):
        self.denoiser.train()
        for epoch in range(epochs):
            avg_loss = 0.0
            k = 0
            for idx, (x0, _) in enumerate(loader):
                x0 = x0.to(self.device, non_blocking=True)
                bs = x0.shape[0]
                t = torch.randint(0, self.timesteps, size=(bs,), dtype=torch.long, device=self.device)
                noise = torch.randn_like(x0)
                noisy_x0 = self.q_forward(x0, noise, t)
                pred_noise = self.denoiser(noisy_x0, t)
                loss = F.mse_loss(pred_noise, noise, reduction='mean')
                avg_loss += loss
                k += 1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if use_wandb and (idx % log_freq) == 0:
                    wandb.log({"train_loss": loss.item()}, step=self.step)
                self.step += 1
            print(f'Epoch {epoch+1}/{epochs}: [Avg loss: {avg_loss.item() / k}]')
            self.epoch += 1
            scheduler.step()

    @torch.no_grad()
    def sample(self, clamp=False, shape=(10, 3, 64, 64)):
        self.denoiser.eval()
        x = torch.randn(shape, device=self.device)
        for idx in tqdm(reversed(range(0, self.timesteps))):
            t = idx * torch.ones((x.shape[0],), device=self.device, dtype=torch.long)
            if idx > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            pred_noise = self.denoiser(x, t)
            x = self.denoise_step(x, pred_noise, t, z)
            if clamp:
                x = x.clamp(-1, 1)
        return x
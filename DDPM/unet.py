import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, parse_shape
import math


# inspired by https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/resnet.py#L45
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, groups=32, dropout=0.1, attn=False):
        super().__init__()
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.temb_fc = nn.Linear(temb_channels, out_channels * 2)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        if attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        t = self.act(self.temb_fc(temb))[:, :, None, None]
        scale, shift = torch.chunk(t, 2, dim=1)
        h = h * (1 + scale) + shift # ref https://github.com/lucidrains/denoising-diffusion-pytorch/issues/77
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return self.attn(h)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, groups=32, dropout=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels, 1)
        self.K = nn.Conv2d(in_channels, in_channels, 1)
        self.V = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.dropout = dropout

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.Q(h), 'b c h w -> b (h w) c')
        k = rearrange(self.K(h), 'b c h w -> b (h w) c')
        v = rearrange(self.V(h), 'b c h w -> b (h w) c')
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        out = rearrange(out, 'b (h w) c -> b c h w', **parse_shape(x, 'b c h w'))
        return x + self.proj(out)


# modification of https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/embeddings.py#L27
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, downscale_freq_shift: float = 0, max_period: int = 10000):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Unet(nn.Module):
    def __init__(self, in_channels = 3, groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = (32, 64, 128, 256)
        self.att_channels = (0, 0, 1, 0)
        self.start_dim = self.out_channels[0]
        self.time_embed_dim = self.start_dim * 4
        self.conv_input = nn.Conv2d(in_channels, self.start_dim, 3, 1, 1)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.start_dim, self.time_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim, bias=True)
        )
        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])
        self._make_paths()
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=2*self.start_dim),
            nn.SiLU(),
            nn.Conv2d(2*self.start_dim, in_channels, 3, 1, 1)
        )

    def forward(self, x, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        timesteps = get_timestep_embedding(timesteps, self.start_dim)
        temb = self.time_embedding(timesteps)
        x = self.conv_input(x)
        h = x.clone()
        down_path = []
        for b1, b2, downsample in self.down:
            h = b1(h, temb)
            down_path.append(h)
            h = b2(h, temb)
            down_path.append(h)
            h = downsample(h)
        h = self.mid[0](h, temb)
        h = self.mid[1](h, temb)
        h = self.mid[2](h)
        for b1, b2, upsample in self.up:
            h = b1(torch.cat((h, down_path.pop()), dim=1), temb)
            h = b2(torch.cat((h, down_path.pop()), dim=1), temb)
            h = upsample(h)
        x = torch.cat((h, x), dim=1)
        return self.final(x)

    def _make_paths(self):
        in_out = list(zip(self.out_channels[:-1], self.out_channels[1:]))
        idx = 0
        for (nin, nout) in in_out:
            attn = self.att_channels[idx] == 1
            first = idx == 0
            down_res =  nn.ModuleList([
                ResnetBlock(in_channels=nin, out_channels=nin, temb_channels=self.time_embed_dim, attn=attn),
                ResnetBlock(in_channels=nin, out_channels=nin, temb_channels=self.time_embed_dim, attn=attn),
                Downsample(in_channels=nin, out_channels=nout)
            ])
            up_res =  nn.ModuleList([
                ResnetBlock(in_channels=(nout + nin), out_channels=nout, temb_channels=self.time_embed_dim, attn=attn),
                ResnetBlock(in_channels=(nout + nin), out_channels=nout, temb_channels=self.time_embed_dim, attn=attn),
                Upsample(in_channels=nout, out_channels=nin) if not first else nn.Conv2d(nout, nin, 3, padding=1)
            ])
            self.down.append(down_res)
            self.up.insert(0, up_res)
            idx += 1
        mid_res = nn.ModuleList([
            ResnetBlock(in_channels=nout, out_channels=nout, temb_channels=self.time_embed_dim, attn=True),
            ResnetBlock(in_channels=nout, out_channels=nout, temb_channels=self.time_embed_dim, attn=False),
            Upsample(in_channels=nout, out_channels=nout),
        ])
        self.mid = mid_res
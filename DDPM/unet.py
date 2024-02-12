import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, parse_shape
import math


# inspired by https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/resnet.py#L45
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, groups=32, dropout=0.1, attn=False):
        super().__init__()
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        if attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.norm2(h)
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
    def __init__(self, in_channels, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.Q = nn.Conv2d(in_channels, in_channels, 1)
        self.K = nn.Conv2d(in_channels, in_channels, 1)
        self.V = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.Q(h), 'b c h w -> b (h w) c')
        k = rearrange(self.K(h), 'b c h w -> b (h w) c')
        v = rearrange(self.V(h), 'b c h w -> b (h w) c')
        out = F.scaled_dot_product_attention(q, k, v)
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
    def __init__(self, ch=128, ch_mul=(1, 2, 2, 2), att_channels=(0, 0, 1, 0), groups=32):
        super().__init__()
        self.ch = ch
        self.ch_mul = ch_mul
        self.att_channels = att_channels
        assert len(self.att_channels) == len(self.ch_mul), 'Attention bool must be defined for each channel'
        self.time_embed_dim = self.ch * 4
        self.input_proj = nn.Conv2d(3, self.ch, 3, 1, 1)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.ch, self.time_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim, bias=True)
        )
        self.down = nn.ModuleList([])
        self.mid = None
        self.up = nn.ModuleList([])
        self._make_paths()
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=2*self.ch),
            nn.SiLU(),
            nn.Conv2d(2*self.ch, 3, 3, 1, 1)
        )

    def forward(self, x, timesteps):
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        timesteps = get_timestep_embedding(timesteps, self.ch)
        temb = self.time_embedding(timesteps)
        x = self.input_proj(x)
        h = x.clone()
        down_path = []
        for i in range(len(self.down)):
            h = self.down[i][0](h, temb)
            down_path.append(h)
            h = self.down[i][1](h, temb)
            down_path.append(h)
            if i < (len(self.down) - 1): # downsample
                h = self.down[i][2](h)
        h = self.mid[0](h, temb)
        h = self.mid[1](h, temb)
        for i in range(len(self.up)):
            h = self.up[i][0](torch.cat((h, down_path.pop()), dim=1), temb)
            h = self.up[i][1](torch.cat((h, down_path.pop()), dim=1), temb)
            if i < (len(self.down) - 1): # upsample
                h = self.up[i][2](h)
        x = torch.cat((h, x), dim=1)
        return self.final(x)

    def _compute_channels(self, res, down):
        nch_in = self.ch * self.ch_mul[res]
        if down:
            if res == (len(self.ch_mul) - 1):
                nch_out = nch_in
            else:
                nch_out = self.ch * self.ch_mul[res + 1]
            transition = Downsample(nch_in, nch_out)
        else:
            if res == 0:
                nch_out = nch_in
            else:
                nch_out = self.ch * self.ch_mul[res - 1]
            transition = Upsample(nch_in, nch_out)
        return nch_in, transition

    def _make_res(self, res, down):
        attn = self.att_channels[res] == 1
        nch_in, transition = self._compute_channels(res, down)
        if not down: nin = 2 * nch_in
        else: nin = nch_in
        return nn.ModuleList([ResnetBlock(nin, nch_in, self.time_embed_dim, attn=attn),
                              ResnetBlock(nin, nch_in, self.time_embed_dim, attn=attn),
                              transition])

    def _make_paths(self):
        num_res = len(self.ch_mul)
        for res in range(num_res):
            down_blocks = self._make_res(res, down=True)
            up_blocks = self._make_res(res, down=False)
            if res == (num_res - 1):
                down_blocks = down_blocks[:-1]
            if res == 0:
                up_blocks = up_blocks[:-1]
            self.down.append(down_blocks)
            self.up.insert(0, up_blocks)
        nch = self.ch * self.ch_mul[-1]
        self.mid = nn.ModuleList([
            ResnetBlock(nch, nch, self.time_embed_dim, attn=True),
            ResnetBlock(nch, nch, self.time_embed_dim, attn=False),
        ])
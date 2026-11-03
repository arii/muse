"""Diffusion policy for the 3-link grasping arm — local, no API key.

A diffusion-type behavior-cloning policy: given the current
observation, it denoises a chunk of future joint-target actions.
This is the state-based Diffusion Policy idea (Chi et al.) with the
UNet replaced by a small MLP denoiser — appropriate for the 16-dim
flat observation (no images), and CPU-friendly.

- Denoiser: MLP on [noisy action chunk (flat), obs, sinusoidal
  timestep embedding] -> predicted noise, same shape as the chunk.
- Noise schedule: diffusers DDPMScheduler for training, DDIMScheduler
  for fast inference (fewer denoising steps).
- Actions are joint targets normalized to [-1, 1] (joint limits / 2.6,
  same convention as the gymnasium wrapper); the sampled chunk is
  clipped to [-1, 1] before execution.
- Observations are normalized with demo mean/std (stored with the
  model); actions are NOT re-normalized beyond [-1, 1].

Execution is receding-horizon: sample a chunk of C actions, execute
the first K, re-observe, replan (see scripts/eval_diffusion.py).

Two denoiser architectures (see --arch in train_diffusion.py):
- "mlp" (legacy default): concat-conditioned MLP. Underfits chunk
  prediction (49% relative error on training points, hedges to the
  mean) — kept for old checkpoints only.
- "unet": FiLM-conditioned 1D temporal UNet (Chi et al. style, light):
  conv down/upsampling over the chunk horizon with obs+timestep
  scale/shift conditioning. Stronger temporal + conditioning bias.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps, dim):
    """Sinusoidal timestep embedding, (B,) long -> (B, dim) float."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                        device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLMBlock(nn.Module):
    """Two convs + GroupNorm with FiLM scale/shift from cond."""

    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.film = nn.Linear(cond_dim, 2 * c_out)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.act(h)


class FiLMUNet1D(nn.Module):
    """FiLM-conditioned 1D temporal UNet denoiser.

    (noisy chunk (B, C, 3), obs (B, obs_dim), t (B,)) -> noise,
    same shape as the chunk. Chunk length must divide by 4.
    """

    def __init__(self, obs_dim, chunk=16, action_dim=3, hidden=64,
                 t_dim=64):
        super().__init__()
        assert chunk % 4 == 0, f"chunk {chunk} must divide by 4"
        self.chunk = chunk
        self.action_dim = action_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, 128), nn.SiLU(),
                                   nn.Linear(128, 128))
        self.o_mlp = nn.Sequential(nn.Linear(obs_dim, 128), nn.SiLU(),
                                   nn.Linear(128, 128))
        cond = 256
        self.down1 = FiLMBlock(action_dim, hidden, cond)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = FiLMBlock(hidden, hidden * 2, cond)
        self.pool2 = nn.MaxPool1d(2)
        self.mid = FiLMBlock(hidden * 2, hidden * 2, cond)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.upb2 = FiLMBlock(hidden * 4, hidden, cond)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.upb1 = FiLMBlock(hidden * 2, hidden, cond)
        self.out = nn.Conv1d(hidden, action_dim, 3, padding=1)

    def forward(self, noisy_actions, obs, timesteps):
        t_emb = self.t_mlp(sinusoidal_embedding(timesteps, 64))
        o_emb = self.o_mlp(obs)
        cond = torch.cat([t_emb, o_emb], dim=-1)
        x = noisy_actions.transpose(1, 2)  # (B, 3, C)
        d1 = self.down1(x, cond)
        d2 = self.down2(self.pool1(d1), cond)
        m = self.mid(self.pool2(d2), cond)
        u2 = self.upb2(torch.cat([self.up2(m), d2], dim=1), cond)
        u1 = self.upb1(torch.cat([self.up1(u2), d1], dim=1), cond)
        return self.out(u1).transpose(1, 2)  # (B, C, 3)


def build_denoiser(ckpt):
    """Construct the denoiser a checkpoint was trained with."""
    arch = ckpt.get("arch", "mlp")
    cls = FiLMUNet1D if arch == "unet" else DiffusionMLP
    model = cls(ckpt["obs_dim"], chunk=ckpt["chunk"],
                hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


class DiffusionMLP(nn.Module):
    """Conditional MLP denoiser: (noisy chunk, obs, t) -> noise."""

    def __init__(self, obs_dim, chunk=16, action_dim=3, hidden=256,
                 depth=3, t_dim=64):
        super().__init__()
        self.chunk = chunk
        self.action_dim = action_dim
        self.t_dim = t_dim
        inp = chunk * action_dim + obs_dim + t_dim
        layers = []
        d = inp
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, chunk * action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_actions, obs, timesteps):
        """noisy_actions: (B, chunk, action_dim); obs: (B, obs_dim);
        timesteps: (B,) long. Returns predicted noise, same shape."""
        b = noisy_actions.shape[0]
        t_emb = sinusoidal_embedding(timesteps, self.t_dim)
        x = torch.cat([noisy_actions.reshape(b, -1), obs, t_emb], dim=-1)
        return self.net(x).reshape(b, self.chunk, self.action_dim)


@torch.no_grad()
def sample_chunk(model, scheduler, obs, chunk, action_dim=3,
                 num_steps=20, device="cpu"):
    """DDIM-sample one action chunk conditioned on obs.

    obs: (obs_dim,) normalized tensor. Returns (chunk, action_dim)
    tensor in [-1, 1]."""
    model.eval()
    scheduler.set_timesteps(num_steps)
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(chunk, action_dim, generator=gen, device=device)
    obs_b = obs.unsqueeze(0).to(device)
    for t in scheduler.timesteps:
        t_b = torch.full((1,), int(t), dtype=torch.long, device=device)
        noise = model(x.unsqueeze(0), obs_b, t_b)[0]
        x = scheduler.step(noise, int(t), x).prev_sample.clamp(-1.2, 1.2)
    return x.clamp(-1.0, 1.0)

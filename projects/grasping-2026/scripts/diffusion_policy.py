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
    x = torch.randn(chunk, action_dim, device=device)
    obs_b = obs.unsqueeze(0).to(device)
    for t in scheduler.timesteps:
        t_b = torch.full((1,), int(t), dtype=torch.long, device=device)
        noise = model(x.unsqueeze(0), obs_b, t_b)[0]
        x = scheduler.step(noise, int(t), x).prev_sample
    return x.clamp(-1.0, 1.0)

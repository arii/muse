"""Train the diffusion policy from collected demos — local, no API.

Samples (observation, action chunk) pairs from demos.npz, trains the
MLP denoiser to predict the DDPM noise, and saves the model + obs
normalization stats + config to models/ (gitignored, binary).

Usage:
    MUJOCO_GL=osmesa python scripts/train_diffusion.py \
        --demos data/demos_envelop --epochs 50 --out models/diffusion
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.diffusion_policy import DiffusionMLP  # noqa: E402


class ChunkDataset:
    """(obs, action chunk) sampler over padded demo arrays.

    Pairs are sliding windows plus FC-anchored windows: for episodes
    with a force-closure step, windows ending exactly at that step
    (repeated anchor_repeat times) so the pinch transition — the
    thing the policy must learn — isn't drowned by approach motion.
    fc_steps: per-episode FC step index (-1 if none), or None.
    """

    def __init__(self, obs, actions, lengths, chunk, fc_steps=None,
                 anchor_repeat=4):
        self.obs, self.actions, self.lengths = obs, actions, lengths
        self.chunk = chunk
        # Valid (episode, start) pairs: chunk must fit in the episode.
        self.pairs = [(i, s) for i, L in enumerate(lengths)
                      for s in range(max(1, L - chunk + 1))]
        if fc_steps is not None:
            for i, fs in enumerate(fc_steps):
                if fs < 0:
                    continue
                # Window ending at the FC transition (start clamped
                # to 0; padding holds the terminal configuration).
                start = max(0, fs - chunk + 1)
                self.pairs += [(i, start)] * anchor_repeat

    def sample(self, n):
        idx = np.random.randint(0, len(self.pairs), size=n)
        o = np.stack([self.obs[i, s] for i, s in
                      [self.pairs[j] for j in idx]])
        a = np.stack([self.actions[i, s:s + self.chunk] for i, s in
                      [self.pairs[j] for j in idx]])
        return (torch.from_numpy(o).float(),
                torch.from_numpy(a).float())


def steps_per_epoch(n_pairs, batch, min_steps):
    """Optimizer steps per epoch: at least min_steps, so a small demo
    set isn't starved (e.g. 135 pairs // batch 64 = 2 steps/epoch
    meant 50 epochs = 100 gradient steps and the loss never moved).
    Sampling is with replacement (ChunkDataset.sample), so a floor
    above n_pairs // batch just reuses pairs within the epoch."""
    return max(min_steps, n_pairs // batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-steps-per-epoch", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="models/diffusion")
    args = ap.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    data = np.load(pathlib.Path(args.demos).with_suffix(".npz"))
    obs, actions, lengths = data["obs"], data["actions"], data["lengths"]
    fc_steps = data["fc_steps"] if "fc_steps" in data else None
    print(f"[train] {len(lengths)} episodes, obs_dim={obs.shape[2]}, "
          f"mean len={lengths.mean():.0f}")
    # Obs stats over true (unpadded) timesteps only.
    # Floor obs_std at 0.30 (and 2.0 on qvel indices 3..5) so
    # near-constant dimensions or transient velocity spikes do not
    # blow up normalized obs during closed-loop rollouts.
    mask = np.arange(obs.shape[1])[None, :] < lengths[:, None]
    flat = obs[mask]
    obs_mean = flat.mean(axis=0)
    obs_std = np.maximum(flat.std(axis=0), 0.30)
    obs_std[3:6] = np.maximum(obs_std[3:6], 2.0)
    obs_n = (obs - obs_mean) / obs_std

    ds = ChunkDataset(obs_n, actions, lengths, args.chunk,
                      fc_steps=fc_steps, anchor_repeat=20)
    if not ds.pairs:
        raise SystemExit("[train] no valid (episode, chunk) pairs — "
                         "episodes shorter than chunk?")
    model = DiffusionMLP(obs.shape[2], chunk=args.chunk,
                         hidden=args.hidden)
    sched = DDPMScheduler(num_train_timesteps=100, clip_sample=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = steps_per_epoch(len(ds.pairs), args.batch,
                            args.min_steps_per_epoch)
    print(f"[train] {len(ds.pairs)} pairs, {steps} steps/epoch")
    for ep in range(args.epochs):
        tot = 0.0
        for _ in range(steps):
            o, a = ds.sample(args.batch)
            o_aug = o + 0.03 * torch.randn_like(o)
            noise = torch.randn_like(a)
            t = torch.randint(0, 100, (args.batch,))
            noisy = sched.add_noise(a, noise, t)
            pred = model(noisy, o_aug, t)
            w_t = 1.0 + 3.0 * (1.0 - t.float() / 100.0)[:, None, None]
            loss = (w_t * (pred - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep}: loss {tot / steps:.4f}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(),
                "obs_mean": obs_mean, "obs_std": obs_std,
                "obs_dim": obs.shape[2], "chunk": args.chunk,
                "hidden": args.hidden},
               str(out.with_suffix(".pt")))
    print(f"[train] saved to {out}.pt")


if __name__ == "__main__":
    main()

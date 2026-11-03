"""Train a single-step MSE behavior-cloning policy — local, no API.

Plain MLP baseline next to PPO and diffusion: obs -> joint-target
action in [-1, 1], trained with MSE on demo (obs, action) pairs.
Closed loop runs one forward pass per env step (see the "bc" planner
in compare_planners.py).

Obs normalization mirrors train_diffusion.py (demo mean/std with a
0.30 floor, 2.0 on qvel) so stats stay comparable across policies.

Usage:
    MUJOCO_GL=osmesa python scripts/train_bc.py \
        --demos data/demos_cage_fc --epochs 150 --out models/bc_v1
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


class BCMLP(nn.Module):
    """Single-step cloning net: obs -> joint targets in [-1, 1]."""

    def __init__(self, obs_dim, action_dim=3, hidden=256, depth=3):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="models/bc_v1")
    args = ap.parse_args()
    torch.manual_seed(0)

    data = np.load(pathlib.Path(args.demos).with_suffix(".npz"))
    obs, actions, lengths = data["obs"], data["actions"], data["lengths"]
    O = np.concatenate([obs[i, : int(lengths[i])]
                        for i in range(len(lengths))]).astype(np.float32)
    A = np.concatenate([actions[i, : int(lengths[i])]
                        for i in range(len(lengths))]).astype(np.float32)
    print(f"[train] {len(lengths)} episodes, {len(O)} pairs, "
          f"obs_dim={O.shape[1]}")
    obs_mean = O.mean(axis=0)
    obs_std = np.maximum(O.std(axis=0), 0.30)
    obs_std[3:6] = np.maximum(obs_std[3:6], 2.0)
    On = torch.from_numpy((O - obs_mean) / obs_std)
    A = torch.from_numpy(A)

    model = BCMLP(O.shape[1], hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = max(50, len(O) // args.batch)
    for ep in range(args.epochs):
        tot = 0.0
        for _ in range(steps):
            idx = torch.randperm(len(On))[:args.batch]
            loss = ((model(On[idx]) - A[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep}: loss {tot / steps:.5f}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "obs_mean": obs_mean,
                "obs_std": obs_std, "obs_dim": O.shape[1],
                "hidden": args.hidden},
               out.with_suffix(".pt"))
    print(f"[train] saved to {out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()

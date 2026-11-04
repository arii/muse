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


def train_val_split(n_episodes, val_frac, seed=0):
    """Deterministic episode indices split (seeded shuffle, then cut).

    Split is by EPISODE, not pair: pairs within an episode are
    temporally correlated, so a pair split would leak. Seeded
    shuffle (not contiguous cut) so scene-blocked files still
    mix across the split.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_episodes)
    n_val = int(round(n_episodes * val_frac))
    return order[n_val:].tolist(), order[:n_val].tolist()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="models/bc_v1")
    args = ap.parse_args(argv)
    torch.manual_seed(0)

    data = np.load(pathlib.Path(args.demos).with_suffix(".npz"))
    obs, actions, lengths = data["obs"], data["actions"], data["lengths"]
    train_idx, val_idx = train_val_split(len(lengths), args.val_frac)
    if not val_idx:  # --val-frac 0: select on train (degenerate, no crash)
        val_idx = train_idx
    O = np.concatenate([obs[i, : int(lengths[i])]
                        for i in train_idx]).astype(np.float32)
    A = np.concatenate([actions[i, : int(lengths[i])]
                        for i in train_idx]).astype(np.float32)
    Ov = np.concatenate([obs[i, : int(lengths[i])]
                         for i in val_idx]).astype(np.float32)
    Av = np.concatenate([actions[i, : int(lengths[i])]
                         for i in val_idx]).astype(np.float32)
    print(f"[train] {len(train_idx)} train / {len(val_idx)} val episodes, "
          f"{len(O)} train pairs, obs_dim={O.shape[1]}")
    obs_mean = O.mean(axis=0)
    obs_std = np.maximum(O.std(axis=0), 0.30)
    obs_std[3:6] = np.maximum(obs_std[3:6], 2.0)
    On = torch.from_numpy((O - obs_mean) / obs_std)
    A = torch.from_numpy(A)
    Ovn = torch.from_numpy((Ov - obs_mean) / obs_std)
    Av = torch.from_numpy(Av)

    model = BCMLP(O.shape[1], hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = max(50, len(O) // args.batch)
    best_val, best_epoch, best_state = float("inf"), -1, None
    for ep in range(args.epochs):
        tot = 0.0
        for _ in range(steps):
            idx = torch.randperm(len(On))[:args.batch]
            loss = ((model(On[idx]) - A[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        with torch.no_grad():
            val = ((model(Ovn) - Av) ** 2).mean().item()
        if val < best_val:
            best_val, best_epoch = val, ep
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep}: train {tot / steps:.5f} "
                  f"val {val:.5f} (best {best_val:.5f} @ {best_epoch})")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "obs_mean": obs_mean,
                "obs_std": obs_std, "obs_dim": O.shape[1],
                "hidden": args.hidden, "best_val": best_val,
                "best_epoch": best_epoch, "val_frac": args.val_frac},
               out.with_suffix(".pt"))
    print(f"[train] saved best (val {best_val:.5f} @ epoch {best_epoch}) "
          f"to {out.with_suffix('.pt')}")


if __name__ == "__main__":
    main()

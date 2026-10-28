"""Evaluate a trained diffusion policy — same headlines as eval_policy.

Receding horizon: sample a chunk of C actions with DDIM, execute the
first K through the gymnasium wrapper, re-observe, replan. Reports
force_closure rate, caged rate, mean reward, mean length. The wrapper
config (--n-objects, --contact-obs) must match training, or the
model's obs dim won't fit.

Usage:
    MUJOCO_GL=osmesa python scripts/eval_diffusion.py \
        --model models/diffusion --episodes 10 --seeds 0-9
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper  # noqa: E402
from scripts.diffusion_policy import DiffusionMLP, sample_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--n-objects", type=int, default=1)
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--max-delta-q", type=float, default=None)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=20)
    ap.add_argument("--gif", default=None)
    args = ap.parse_args()

    ckpt = torch.load(pathlib.Path(args.model).with_suffix(".pt"),
                      map_location="cpu", weights_only=False)
    model = DiffusionMLP(ckpt["obs_dim"], chunk=ckpt["chunk"],
                         hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    sched = DDIMScheduler(num_train_timesteps=100)
    obs_mean = torch.from_numpy(ckpt["obs_mean"]).float()
    obs_std = torch.from_numpy(ckpt["obs_std"]).float()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    seeds = seeds[:args.episodes]

    fc = caged = 0
    rewards, lengths = [], []
    for i, seed in enumerate(seeds):
        env = GraspGymWrapper(seed=seed, max_steps=args.max_steps,
                              n_objects=args.n_objects,
                              contact_obs=args.contact_obs,
                              max_delta_q=args.max_delta_q)
        obs, _ = env.reset(seed=seed)
        frames = []
        total, steps = 0.0, 0
        info = {}
        while True:
            obs_n = (torch.from_numpy(obs).float() - obs_mean) / obs_std
            chunk = sample_chunk(model, sched, obs_n, ckpt["chunk"],
                                 num_steps=args.ddim_steps)
            for k in range(min(args.exec_horizon, len(chunk))):
                obs, reward, terminated, truncated, info = env.step(
                    chunk[k].numpy())
                total += reward
                steps += 1
                if args.gif and i == 0:
                    frames.append(env.render())
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        fc += int(info["force_closure"])
        caged += int(info["caged"])
        rewards.append(total)
        lengths.append(steps)
        print(f"seed {seed:>3} FC={info['force_closure']!s:5} "
              f"caged={info['caged']!s:5} rew={total:.1f} len={steps}")
        if args.gif and i == 0 and frames:
            from scripts.make_video import make_gif
            make_gif(np.stack(frames),
                     {"force_closure": fc > 0}, args.gif,
                     fps=15, scale=1.0)
            print(f"[eval] gif -> {args.gif}")

    n = len(seeds)
    print(f"== diffusion ({n} episodes) ==")
    print(f"  force_closure rate: {fc / n:.2f}")
    print(f"  caged rate:         {caged / n:.2f}")
    print(f"  mean reward:        {np.mean(rewards):.1f}")
    print(f"  mean length:        {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()

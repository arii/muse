"""Evaluate a trained grasping policy — the comparison surface for the
trained approach vs VLM and heuristic (which report force_closure rate
in compare_planners.py).

Runs the policy deterministically for N episodes and reports
force_closure rate, caged rate, mean reward, and mean episode length.
Optionally saves a GIF of the first episode.

Usage:
    MUJOCO_GL=osmesa python scripts/eval_policy.py --model models/grasp_ppo
    MUJOCO_GL=osmesa python scripts/eval_policy.py --model models/grasp_ppo \
        --episodes 10 --seeds 0-9 --gif /tmp/trained_seed0.gif
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--n-objects", type=int, default=4)
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--max-delta-q", type=float, default=None)
    ap.add_argument("--gif", default=None)
    args = ap.parse_args()

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        cls = PPO
    else:
        from stable_baselines3 import SAC
        cls = SAC
    model = cls.load(args.model)

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    seeds = seeds[:args.episodes]

    fc = caged = 0
    rewards, lengths = [], []
    gif_frames = None
    for i, seed in enumerate(seeds):
        env = GraspGymWrapper(seed=seed, max_steps=args.max_steps,
                              n_objects=args.n_objects,
                              contact_obs=args.contact_obs,
                              max_delta_q=args.max_delta_q)
        obs, _ = env.reset(seed=seed)
        frames = []
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if args.gif and i == 0:
                frames.append(env.render())
            if terminated or truncated:
                break
        fc += int(info["force_closure"])
        caged += int(info["caged"])
        rewards.append(total)
        lengths.append(steps)
        print(f"seed {seed:>3} FC={info['force_closure']!s:5} "
              f"caged={info['caged']!s:5} rew={total:.1f} len={steps}")
        if args.gif and i == 0:
            gif_frames = frames

    n = len(seeds)
    print(f"== trained ({args.algo}, {n} episodes) ==")
    print(f"  force_closure rate: {fc / n:.2f}")
    print(f"  caged rate:         {caged / n:.2f}")
    print(f"  mean reward:        {np.mean(rewards):.1f}")
    print(f"  mean length:        {np.mean(lengths):.0f}")
    if gif_frames:
        from scripts.make_video import make_gif
        meta = {"force_closure": fc > 0}
        make_gif(np.stack(gif_frames), meta, args.gif, fps=15, scale=1.0)
        print(f"[eval] gif -> {args.gif}")


if __name__ == "__main__":
    main()

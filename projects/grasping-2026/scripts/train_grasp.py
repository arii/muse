"""Train a grasping policy locally (CPU, no API key).

PPO (default) or SAC on the gymnasium-wrapped WholeArmGraspEnv.
Reward is the env's own: 0.5 * links touching + 2.0 * force_closure.
Success (episode termination) is force_closure.

Usage:
    MUJOCO_GL=osmesa python scripts/train_grasp.py --timesteps 100000
    MUJOCO_GL=osmesa python scripts/train_grasp.py --algo sac \
        --timesteps 50000 --out models/grasp_sac
    # then evaluate:
    MUJOCO_GL=osmesa python scripts/eval_policy.py --model models/grasp_ppo

No network, no API key: MuJoCo + torch run locally. CPU-only here
(torch reports cuda=False on this VM); a GPU just speeds it up.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--out", default="models/grasp_ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--target-index", type=int, default=0)
    args = ap.parse_args()

    if args.algo == "ppo":
        from stable_baselines3 import PPO
        cls = PPO
    else:
        from stable_baselines3 import SAC
        cls = SAC

    env = GraspGymWrapper(seed=args.seed, target_index=args.target_index,
                          max_steps=args.max_steps)
    model = cls("MlpPolicy", env, seed=args.seed, verbose=1)
    print(f"[train] {args.algo} for {args.timesteps} timesteps "
          f"(max_steps={args.max_steps}, target={args.target_index})")
    model.learn(total_timesteps=args.timesteps)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"[train] saved to {out}.zip")


if __name__ == "__main__":
    main()

"""Collect demonstrations from a trained PPO/SAC policy — no API.

Behavior cloning cannot exceed its demonstrator, and neither scripted
demonstrator (cage, envelop) ever achieves force closure (0/10 seeds
each). The 500k-step PPO policy DOES (seed 9, 2 steps). So this script
rolls out a trained policy in the gymnasium wrapper and records
(obs, action) per step, keeping episodes that meet --keep. Output
format matches scripts/collect_demos.py, so train_diffusion.py works
unchanged.

Caveat: FC episodes are short (the episode ends on force closure),
so chunks from them are mostly hold-last padding — the demo says
"reach this configuration and stay", which is the right semantics
for a pinch grasp.

Usage:
    MUJOCO_GL=osmesa python scripts/collect_ppo_demos.py \
        --model models/grasp_ppo_500k --seeds 0-99 --keep fc \
        --out data/demos_ppo_fc
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--n-objects", type=int, default=4)
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--max-delta-q", type=float, default=None)
    ap.add_argument("--keep", choices=["fc", "caged", "all"],
                    default="fc")
    ap.add_argument("--out", default="data/demos_ppo")
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

    kept_obs, kept_act, kept_len = [], [], []
    kept_fc_step = []
    n_fc = n_caged = 0
    for seed in seeds:
        env = GraspGymWrapper(seed=seed, max_steps=args.max_steps,
                              n_objects=args.n_objects,
                              contact_obs=args.contact_obs,
                              max_delta_q=args.max_delta_q)
        obs, _ = env.reset(seed=seed)
        obs_list, act_list = [], []
        info = {}
        fc_step = -1
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs_list.append(obs)
            act_list.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            if fc_step < 0 and info["force_closure"]:
                fc_step = len(obs_list) - 1
            if terminated or truncated:
                break
        n_fc += int(info["force_closure"])
        n_caged += int(info["caged"])
        keep = (args.keep == "all" or
                (args.keep == "fc" and info["force_closure"]) or
                (args.keep == "caged" and info["caged"]))
        if keep:
            kept_obs.append(np.stack(obs_list))
            kept_act.append(np.stack(act_list))
            kept_len.append(len(obs_list))
            kept_fc_step.append(fc_step)
        print(f"seed {seed}: FC={info['force_closure']} "
              f"caged={info['caged']} len={len(obs_list)} kept={keep}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if kept_obs:
        maxlen = max(kept_len)
        obs_dim = kept_obs[0].shape[1]
        obs_pad = np.zeros((len(kept_obs), maxlen, obs_dim),
                           dtype=np.float32)
        act_pad = np.zeros((len(kept_obs), maxlen, 3), dtype=np.float32)
        for i, (o, a) in enumerate(zip(kept_obs, kept_act)):
            obs_pad[i, :len(o)] = o
            act_pad[i, :len(a)] = a
            # Hold-last padding (not zeros): a chunk past the end
            # means "stay in the terminal configuration".
            obs_pad[i, len(o):] = o[-1]
            act_pad[i, len(a):] = a[-1]
        np.savez_compressed(out.with_suffix(".npz"), obs=obs_pad,
                            actions=act_pad,
                            lengths=np.array(kept_len),
                            fc_steps=np.array(kept_fc_step))
    meta = {"demonstrator": f"{args.algo}:{args.model}", "keep": args.keep,
            "n_objects": args.n_objects, "contact_obs": args.contact_obs,
            "seeds": seeds, "episodes_run": len(seeds),
            "episodes_kept": len(kept_obs), "fc_count": n_fc,
            "caged_count": n_caged}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] kept {len(kept_obs)}/{len(seeds)} episodes "
          f"(FC {n_fc}, caged {n_caged}) -> {out}.npz")


if __name__ == "__main__":
    main()

"""Collect scripted demonstrations for diffusion training — no API.

Runs a scripted demonstrator (cage or envelop) on the base env and
records (observation, action) per step, where observations are the
gymnasium wrapper's flat obs (so the diffusion policy sees exactly
what it will see at eval) and actions are joint targets normalized
to [-1, 1] (joint limits / 2.6, same as the wrapper).

Two recording tricks keep the demos clean and fast:
- Rendering is patched to a dummy during collection (the demonstrator
  renders every step, but demos don't need frames).
- Only episodes meeting --keep are saved: fc (force closure),
  caged, or all. Behavior cloning cannot exceed its demonstrator, so
  the default keeps force-closure episodes only — if the demonstrator
  never passes, the output is empty and says so plainly.

Output: <out>.npz (obs, actions padded to max length, lengths) plus
<out>.json metadata (demonstrator, keep rule, seeds, counts).

Usage:
    MUJOCO_GL=osmesa python scripts/collect_demos.py \
        --demonstrator envelop --seeds 0-99 --n-objects 1 \
        --out data/demos_envelop
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from env.gym_wrapper import GraspGymWrapper  # noqa: E402
from env.whole_arm_grasp import WholeArmGraspEnv  # noqa: E402


def collect_one(seed, demonstrator, n_objects, contact_obs):
    """Run one demonstrator episode; return (obs, actions, info) or
    None if the demonstrator itself fails."""
    wrapper = GraspGymWrapper(seed=seed, n_objects=n_objects,
                              contact_obs=contact_obs)
    base = wrapper.env
    base.reset(seed=seed)
    base.target_index = 0
    # No rendering during collection (demos don't need frames).
    base.render = lambda: np.zeros((2, 2, 3), dtype=np.uint8)
    _, last_info = base._reward()
    obs_list, act_list = [], []

    orig_step = base.step

    def rec_step(action, substeps=20):
        nonlocal last_info
        obs_list.append(wrapper._flat_obs(last_info))
        act_list.append(np.asarray(action, dtype=float) / 2.6)
        out = orig_step(action, substeps)
        last_info = out[3]
        return out

    base.step = rec_step
    frames = []
    try:
        if demonstrator == "cage":
            from scripts.demo_vlm_grasp import scripted_cage
            _, info = scripted_cage(base, 0, frames)
        else:
            from planning.primitives import envelop
            tpos = base._obs()["objects"][0][:2].copy()
            info = envelop(base, tpos, frames)
    except Exception as e:
        print(f"seed {seed}: demonstrator failed: {e}")
        return None
    return np.stack(obs_list), np.stack(act_list), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demonstrator", choices=["cage", "envelop"],
                    default="envelop")
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--n-objects", type=int, default=1)
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--keep", choices=["fc", "caged", "all"],
                    default="fc")
    ap.add_argument("--out", default="data/demos")
    args = ap.parse_args()

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    kept_obs, kept_act, kept_len, kept_fc = [], [], [], []
    n_fc = n_caged = 0
    for seed in seeds:
        res = collect_one(seed, args.demonstrator, args.n_objects,
                          args.contact_obs)
        if res is None:
            continue
        obs, act, info = res
        n_fc += int(info["force_closure"])
        n_caged += int(info["caged"])
        keep = (args.keep == "all" or
                (args.keep == "fc" and info["force_closure"]) or
                (args.keep == "caged" and info["caged"]))
        if keep:
            kept_obs.append(obs)
            kept_act.append(act)
            kept_len.append(len(obs))
            kept_fc.append(bool(info["force_closure"]))
        print(f"seed {seed}: FC={info['force_closure']} "
              f"caged={info['caged']} len={len(obs)} "
              f"kept={keep}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if kept_obs:
        maxlen = max(kept_len)
        obs_dim = kept_obs[0].shape[1]
        obs_pad = np.zeros((len(kept_obs), maxlen, obs_dim),
                           dtype=np.float32)
        act_pad = np.zeros((len(kept_obs), maxlen, 3), dtype=np.float32)
        fc_steps = np.array([
            L - 1 if f else -1
            for L, f in zip(kept_len, kept_fc)
        ], dtype=np.int64)
        for i, (o, a) in enumerate(zip(kept_obs, kept_act)):
            obs_pad[i, :len(o)] = o
            act_pad[i, :len(a)] = a
            # Hold-last padding (not zeros): a chunk past the end
            # means "stay in the terminal configuration".
            obs_pad[i, len(o):] = o[-1]
            act_pad[i, len(a):] = a[-1]
        # Extend lengths by 16 hold-last steps so ChunkDataset samples
        # post-FC hold chunks and the policy finishes the final pinch.
        ext_len = np.minimum(np.array(kept_len) + 16, maxlen)
        obs_pad_ext = np.zeros((len(kept_obs), maxlen + 16, obs_dim),
                               dtype=np.float32)
        act_pad_ext = np.zeros((len(kept_obs), maxlen + 16, 3),
                               dtype=np.float32)
        for i, (o, a) in enumerate(zip(kept_obs, kept_act)):
            obs_pad_ext[i, :len(o)] = o
            act_pad_ext[i, :len(a)] = a
            obs_pad_ext[i, len(o):] = o[-1]
            act_pad_ext[i, len(a):] = a[-1]
        np.savez_compressed(out.with_suffix(".npz"), obs=obs_pad_ext,
                            actions=act_pad_ext,
                            lengths=np.array(kept_len) + 16,
                            fc_steps=fc_steps)
    meta = {"demonstrator": args.demonstrator, "keep": args.keep,
            "n_objects": args.n_objects, "contact_obs": args.contact_obs,
            "seeds": seeds, "episodes_run": len(seeds),
            "episodes_kept": len(kept_obs), "fc_count": n_fc,
            "caged_count": n_caged}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] kept {len(kept_obs)}/{len(seeds)} episodes "
          f"(FC {n_fc}, caged {n_caged}) -> {out}.npz")


if __name__ == "__main__":
    main()

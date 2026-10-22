"""Demo loop: instruction -> VLM grounding -> scripted caging -> logged episode.

Usage:
    python scripts/demo_vlm_grasp.py --instruction "grasp the red box"

The controller here is SCRIPTED (a placeholder). It demonstrates the full
loop; replace it with a learned policy (see train_rl_baseline.py) or a
VLM-driven motion plan. Episodes land in episodes/<timestamp>/.
"""
import argparse
import datetime
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from env.whole_arm_grasp import WholeArmGraspEnv
from perception.grounding import make_labeller


def scripted_cage(env, target_idx, frames):
    """Three-phase scripted whole-arm cage around the target object."""
    obs = env._obs()
    pos = obs["objects"][target_idx][:2]
    face = float(np.arctan2(pos[1], pos[0]))
    phases = [
        ([face, 0.15, 0.15], 120),    # 1. yaw toward the object, arm extended
        ([face, 0.55, 0.55], 120),    # 2. reach: bend elbow toward it
        ([face, 1.25, 1.45], 200),    # 3. curl: wrap links around it
    ]
    rewards = []
    for angles, ticks in phases:
        for _ in range(ticks // 20):
            _, r, _, info = env.step(np.array(angles))
            rewards.append(r)
            frames.append(env.render())
            if info["caged"]:
                return rewards, info
    _, r, _, info = env.step(np.array(phases[-1][0]))
    rewards.append(r)
    return rewards, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="grasp the red box")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = WholeArmGraspEnv(seed=args.seed)
    env.reset(seed=args.seed)
    frames = [env.render()]

    labeller = make_labeller()
    target_idx, rationale = labeller.label(
        frames[0], env.objects, args.instruction)
    env.target_index = target_idx
    print(f"[demo] instruction: {args.instruction!r}")
    print(f"[demo] labeller chose object {target_idx} "
          f"({env.objects[target_idx]['color']}) — {rationale}")

    rewards, info = scripted_cage(env, target_idx, frames)
    print(f"[demo] links touching: {info['links_touching']}, "
          f"caged: {info['caged']}, total reward: {sum(rewards):.1f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("episodes") / stamp
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz",
                        frames=np.stack(frames)[::4])  # subsampled
    (out / "meta.json").write_text(json.dumps({
        "instruction": args.instruction,
        "target_index": target_idx,
        "target_color": env.objects[target_idx]["color"],
        "rationale": rationale,
        "links_touching": info["links_touching"],
        "caged": info["caged"],
        "total_reward": float(sum(rewards)),
    }, indent=2))
    print(f"[demo] episode saved to {out}")


if __name__ == "__main__":
    main()

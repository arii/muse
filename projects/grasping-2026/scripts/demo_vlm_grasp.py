"""Demo loop: instruction -> VLM grounding -> singulation -> scripted caging.

Usage:
    python scripts/demo_vlm_grasp.py --instruction "grasp the red box"

Singulation (new): before caging, the arm shoves aside any non-target
objects crowding the target, isolating it. The controller here is SCRIPTED
(a placeholder). It demonstrates the full loop; replace it with a learned
policy (see train_rl_baseline.py) or a VLM-driven motion plan. Episodes
land in episodes/<timestamp>/.
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


def _ang_diff(a, b):
    """Signed smallest difference a - b, in [-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _tip_reach(q):
    """Approx arm tip radius for j2 = j3 = q (arm straight along j1)."""
    return 0.30 + 0.25 * np.cos(q) + 0.20 * np.cos(2 * q)


def _extension_for(target_reach):
    """j2 = j3 angle giving tip reach ~= target_reach (bisection)."""
    lo, hi = 0.12, 1.4  # _tip_reach decreasing on this interval
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _tip_reach(mid) > target_reach:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scripted_singulate(env, target_idx, frames, push_radius=0.35):
    """Push-away phase: isolate the target before grasping.

    For each non-target object crowding the target (nearest first): curl
    the arm (small swept disc), yaw to face the object, extend just far
    enough for the tip to reach it, then nudge it tangentially in the
    direction that increases its distance from the target. Each push is
    tried optimistically and REVERTED (physics snapshot + dropped frames)
    if it hurts: isolation decreased, the target moved > 5 cm, or any
    object flew > 50 cm. Everything moves slowly — yawing or sweeping the
    fully-extended arm is what launches boxes across the table.
    Returns (isolation_before, isolation_after) in meters.
    """
    def xy(i):
        return env._obs()["objects"][i][:2].copy()

    others = [o["index"] for o in env.objects if o["index"] != target_idx]
    iso_start = env.target_isolation()
    iso_before = iso_start
    done = set()
    while True:
        t = xy(target_idx)
        cands = sorted(
            (float(np.linalg.norm(xy(i) - t)), i) for i in others
            if i not in done and float(np.linalg.norm(xy(i) - t)) < push_radius)
        if not cands:
            break
        _, i = cands[0]
        done.add(i)
        d = xy(i)
        face = float(np.arctan2(d[1], d[0]))
        if abs(face) > 2.5:  # j1 range is +-2.6; can't face it squarely
            continue
        color = next(o["color"] for o in env.objects if o["index"] == i)
        snap = env.snapshot()
        nframes0 = len(frames)
        p0 = {o["index"]: xy(o["index"]) for o in env.objects}
        r_d = float(np.linalg.norm(d))
        q0 = env._obs()["qpos"]
        # 1. curl: shrink the swept disc before any yaw
        for a in np.linspace(0, 1, 10):
            env.step(np.array([q0[0], (1 - a) * q0[1] + a * 1.4,
                               (1 - a) * q0[2] + a * 1.4]))
            frames.append(env.render())
        # 2. yaw while curled (short way around)
        for a in np.linspace(0, 1, 20):
            env.step(np.array([q0[0] + _ang_diff(face, q0[0]) * a, 1.4, 1.4]))
            frames.append(env.render())
        # 3. extend so the tip just reaches the object (no bulldozing through)
        qe = _extension_for(max(r_d - 0.02, 0.2))
        for a in np.linspace(0, 1, 15):
            q23 = 1.4 - (1.4 - qe) * a
            env.step(np.array([face, q23, q23]))
            frames.append(env.render())
        # 4. tangential nudge, signed to push away from the target
        d = xy(i)  # re-read: extension may have shifted it
        that = np.array([-np.sin(face), np.cos(face)])
        s = 1.0 if float(np.dot(that, d - t)) >= 0 else -1.0
        for a in np.linspace(0, 1, 16):
            env.step(np.array([face + s * 0.08 * a, qe, qe]))
            frames.append(env.render())
        # evaluate: keep the push only if it moved the DISTRACTOR away
        # (not the target), without launching anything
        iso_new = env.target_isolation()
        tmove = float(np.linalg.norm(xy(target_idx) - t))
        dmove = float(np.linalg.norm(xy(i) - p0[i]))
        maxmove = max(float(np.linalg.norm(xy(j) - p0[j])) for j in p0)
        if (iso_new < iso_before - 1e-9 or maxmove > 0.6 or dmove < 0.03
                or tmove > dmove + 0.05):
            env.restore(snap)
            del frames[nframes0:]
            print(f"[singulate] reverted push on obj{i} ({color}): "
                  f"iso {iso_before:.3f}->{iso_new:.3f}, "
                  f"distractor moved {dmove:.3f} m, "
                  f"target moved {tmove:.3f} m")
        else:
            iso_before = iso_new
            print(f"[singulate] pushed obj{i} ({color}): "
                  f"iso {iso_new:.3f} m")
    return iso_start, env.target_isolation()


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
    ap.add_argument("--frame-every", type=int, default=4,
                    help="save every Nth frame (1 = full rate, for video)")
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

    iso_before, iso_after = scripted_singulate(env, target_idx, frames)
    print(f"[demo] singulation: isolation {iso_before:.3f} -> "
          f"{iso_after:.3f} m")

    rewards, info = scripted_cage(env, target_idx, frames)
    print(f"[demo] links touching: {info['links_touching']}, "
          f"caged: {info['caged']}, total reward: {sum(rewards):.1f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("episodes") / stamp
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz",
                        frames=np.stack(frames)[::args.frame_every])
    (out / "meta.json").write_text(json.dumps({
        "instruction": args.instruction,
        "target_index": target_idx,
        "target_color": env.objects[target_idx]["color"],
        "rationale": rationale,
        "isolation_before": iso_before,
        "isolation_after": iso_after,
        "links_touching": info["links_touching"],
        "caged": info["caged"],
        "total_reward": float(sum(rewards)),
    }, indent=2))
    print(f"[demo] episode saved to {out}")


if __name__ == "__main__":
    main()

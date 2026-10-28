"""Demo loop: goal -> VLM preconditions -> singulation -> isolation gate -> cage.

The task is "grasp the target IN ISOLATION". The VLM sees the initial scene
and reasons about preconditions (which boxes must be pushed away first).
After the pushes, an isolation gate checks the preconditions are met before
the grasp starts.

Usage:
    python scripts/demo_vlm_grasp.py --instruction "grasp the red box in isolation"

The controller here is SCRIPTED (a placeholder). It demonstrates the full
loop; replace it with a learned policy (see train_rl_baseline.py) or a
VLM-driven motion plan. Episodes land in episodes/<timestamp>/ with
frames.npz, meta.json, and demo.gif (built in — no separate video step).
Use --no-gif to skip the GIF.
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
from planning.primitives import push


def scripted_singulate(env, target_idx, frames, push_radius=0.35,
                      push_targets=None, strict=False):
    """Push-away phase: isolate the target before grasping.

    For each non-target object crowding the target (nearest first), or for
    each index in push_targets when explicitly given: compute a Cartesian
    push (start at the object's target-facing edge, direction away from the
    target) and execute it via planning.primitives.push. Each push is tried
    optimistically and REVERTED (physics snapshot + dropped frames) if it
    hurts: isolation decreased, the target moved more than the distractor
    (or at all, when strict=True e.g. "without moving the red block"), the
    distractor barely moved, or something launched.
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
        if push_targets is not None:
            # explicit instruction: push exactly these, no distance filter
            cands = [(0.0, i) for i in push_targets
                     if i in others and i not in done]
        else:
            cands = sorted(
                (float(np.linalg.norm(xy(i) - t)), i) for i in others
                if i not in done
                and float(np.linalg.norm(xy(i) - t)) < push_radius)
        if not cands:
            break
        _, i = cands[0]
        done.add(i)
        d = xy(i)
        t = xy(target_idx)
        to_d = d - t
        if float(np.linalg.norm(to_d)) < 1e-6:
            continue
        direction = to_d / np.linalg.norm(to_d)
        theta = float(np.arctan2(direction[1], direction[0]))
        # Cartesian push: start at the distractor's target-facing edge
        # (box half-size + tip radius + margin), push directly away
        sx = next(o["half"][0] for o in env.objects if o["index"] == i)
        clearance = sx + 0.025 + (0.010 if strict else 0.015)
        start_xy = d - direction * clearance
        color = next(o["color"] for o in env.objects if o["index"] == i)
        snap = env.snapshot()
        nframes0 = len(frames)
        p0 = {o["index"]: xy(o["index"]) for o in env.objects}
        push(env, start_xy, theta, 0.06 if strict else 0.15, frames)
        # evaluate: keep the push only if it moved the DISTRACTOR away
        # (not the target), without launching anything
        iso_new = env.target_isolation()
        tmove = float(np.linalg.norm(xy(target_idx) - t))
        dmove = float(np.linalg.norm(xy(i) - p0[i]))
        maxmove = max(float(np.linalg.norm(xy(j) - p0[j])) for j in p0)
        tmove_limit = 0.10 if strict else dmove + 0.05
        if (iso_new < iso_before - 1e-9 or maxmove > 0.6 or dmove < 0.03
                or tmove > tmove_limit):
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
        # ticks // 5 (not // 20): the motion is the demo. Fewer steps
        # made the arm snap between poses and the GIF ran in ~1.5 s.
        # More steps = smaller increments, settled actuators, and a
        # watchable ~6 s cage instead of a teleport.
        for _ in range(ticks // 5):
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
    ap.add_argument("--instruction", default="grasp the red box in isolation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-every", type=int, default=4,
                    help="save every Nth frame (1 = full rate, for video)")
    ap.add_argument("--no-gif", action="store_true",
                    help="skip writing demo.gif (GIF is on by default)")
    ap.add_argument("--gif-fps", type=float, default=15,
                    help="GIF frame rate (default 15)")
    args = ap.parse_args()

    env = WholeArmGraspEnv(seed=args.seed)
    env.reset(seed=args.seed)
    frames = [env.render()]

    labeller = make_labeller()
    try:
        plan = labeller.plan(frames[0], env.objects, args.instruction)
    except Exception as e:
        # VLM failed (bad model id, null content, API error, ...) — print
        # the full diagnostic so the failure is debuggable, then fall back
        # to the offline heuristic so the demo still runs.
        print(f"[demo] VLM plan failed — diagnostic:\n{e}")
        print("[demo] falling back to heuristic")
        from perception.grounding import HeuristicLabeller
        plan = HeuristicLabeller().plan(frames[0], env.objects,
                                        args.instruction)
    target_idx = plan["grasp_target"]
    env.target_index = target_idx
    print(f"[demo] instruction: {args.instruction!r}")
    print(f"[demo] plan: grasp object {target_idx} "
          f"({env.objects[target_idx]['color']}), push away "
          f"{plan['push_away']} — {plan['rationale']}")

    # Explicit "push ..." in the instruction -> push exactly those objects.
    # Otherwise fall back to auto-detecting crowding. "without moving"
    # makes the revert strict: any target motion fails the push.
    explicit_push = "push" in args.instruction.lower()
    push_targets = plan["push_away"] if explicit_push else None
    strict = "without moving" in args.instruction.lower()
    iso_before, iso_after = scripted_singulate(
        env, target_idx, frames, push_targets=push_targets, strict=strict)
    print(f"[demo] singulation: isolation {iso_before:.3f} -> "
          f"{iso_after:.3f} m")

    # Isolation gate: the task is "grasp IN ISOLATION". Only start the grasp
    # once preconditions are met (or log that they weren't).
    ISOLATION_THRESHOLD = 0.15  # meters of clearance
    preconditions_met = iso_after >= ISOLATION_THRESHOLD
    if preconditions_met:
        print(f"[demo] preconditions met (isolation {iso_after:.3f} m "
              f">= {ISOLATION_THRESHOLD} m) — starting grasp")
    else:
        print(f"[demo] WARNING: preconditions NOT met (isolation "
              f"{iso_after:.3f} m < {ISOLATION_THRESHOLD} m) — "
              f"attempting grasp anyway")

    rewards, info = scripted_cage(env, target_idx, frames)
    print(f"[demo] links touching: {info['links_touching']}, "
          f"enclosed: {info['enclosed']}, caged: {info['caged']}, "
          f"force_closure: {info['force_closure']}, "
          f"total reward: {sum(rewards):.1f}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("episodes") / stamp
    out.mkdir(parents=True, exist_ok=True)
    saved_frames = np.stack(frames)[::args.frame_every]
    np.savez_compressed(out / "frames.npz", frames=saved_frames)
    meta = {
        "instruction": args.instruction,
        "target_index": target_idx,
        "target_color": env.objects[target_idx]["color"],
        "rationale": plan["rationale"],
        "push_away": plan["push_away"],
        "isolation_before": iso_before,
        "isolation_after": iso_after,
        "preconditions_met": preconditions_met,
        "links_touching": info["links_touching"],
        "caged": info["caged"],
        "force_closure": info["force_closure"],
        "total_reward": float(sum(rewards)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[demo] episode saved to {out}")

    # GIF, built in — no separate make_video.py call needed.
    # Uses EVERY frame (not the subsampled npz frames): the episode is
    # only a few dozen frames, and frame_every=4 turns the GIF into
    # ~10 teleporting poses. Full rate + full scale is what looks good.
    if not args.no_gif:
        from scripts.make_video import make_gif
        gif_path = make_gif(np.stack(frames), meta, out / "demo.gif",
                            fps=args.gif_fps, scale=1.0)
        print(f"[demo] GIF saved to {gif_path}")


if __name__ == "__main__":
    main()

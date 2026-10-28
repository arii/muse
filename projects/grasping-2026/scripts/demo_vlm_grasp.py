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
from planning.primitives import (EFFORT_DISTANCES, approach_face,
                                 frame_status, push)


def normalize_push_targets(push_targets):
    """Plan push_away entries -> [(index, effort)].

    Accepts the canonical dicts and legacy bare indices (effort
    "clear"). Raises on an unknown effort — the planner must pick
    from the schema's enum, the executor must not guess.
    """
    out = []
    for item in push_targets or []:
        if isinstance(item, dict):
            idx, effort = int(item["index"]), item.get("effort", "clear")
        else:
            idx, effort = int(item), "clear"
        if effort not in EFFORT_DISTANCES:
            raise ValueError(f"unknown push effort {effort!r}")
        out.append((idx, effort))
    return out


def scripted_singulate(env, target_idx, frames, push_radius=0.35,
                       push_targets=None, strict=False, frame_info=None,
                       push_log=None):
    """Push-away phase: isolate the target before grasping.

    For each entry in push_targets (in order) when explicitly given, or
    for each non-target object crowding the target (nearest first) when
    push_targets is None: compute a Cartesian push (start at the
    object's target-facing edge, direction away from the target) and
    execute it via planning.primitives.push. The push distance comes
    from the entry's effort (nudge/clear/shove), or 0.06 when strict
    ("without moving" overrides effort: don't disturb). Each push is
    tried optimistically and REVERTED (physics snapshot + dropped
    frames) if it hurts: isolation decreased, the target moved more
    than the distractor (or at all, when strict), the distractor
    barely moved, or something launched. Kept pushes are appended to
    push_log (if given) as {index, effort, distance_m}.
    Returns (isolation_before, isolation_after) in meters.
    """
    def xy(i):
        return env._obs()["objects"][i][:2].copy()

    others = [o["index"] for o in env.objects if o["index"] != target_idx]
    explicit = normalize_push_targets(push_targets) \
        if push_targets is not None else None
    iso_start = env.target_isolation()
    iso_before = iso_start
    done = set()
    while True:
        t = xy(target_idx)
        if explicit is not None:
            # explicit plan: push exactly these, in order, no filter
            cands = [(0.0, i, e) for i, e in explicit
                     if i in others and i not in done]
        else:
            cands = sorted(
                (float(np.linalg.norm(xy(i) - t)), i, "clear")
                for i in others
                if i not in done
                and float(np.linalg.norm(xy(i) - t)) < push_radius)
        if not cands:
            break
        _, i, effort = cands[0]
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
        ninfo0 = len(frame_info) if frame_info is not None else 0
        p0 = {o["index"]: xy(o["index"]) for o in env.objects}
        dist = 0.06 if strict else EFFORT_DISTANCES[effort]
        push(env, start_xy, theta, dist, frames,
             frame_info=frame_info)
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
            if frame_info is not None:
                del frame_info[ninfo0:]
            print(f"[singulate] reverted push on obj{i} ({color}): "
                  f"iso {iso_before:.3f}->{iso_new:.3f}, "
                  f"distractor moved {dmove:.3f} m, "
                  f"target moved {tmove:.3f} m")
        else:
            iso_before = iso_new
            if push_log is not None:
                push_log.append({"index": i, "effort": effort,
                                 "distance_m": dist})
            print(f"[singulate] pushed obj{i} ({color}, {effort}): "
                  f"iso {iso_new:.3f} m")
    return iso_start, env.target_isolation()


def scripted_cage(env, target_idx, frames, frame_info=None,
                  approach_side="near"):
    """Three-phase scripted whole-arm cage around the target object.

    approach_side biases the entry yaw (see approach_face): near is
    the historical behavior (yaw toward the target); far/left/right
    swing the arm in from another side, which matters when a
    distractor sits on the near side.
    """
    obs = env._obs()
    pos = obs["objects"][target_idx][:2]
    face = approach_face(pos, approach_side)
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
            if frame_info is not None:
                frame_info.append(frame_status(info))
            # Stop once the grasp is secured by EITHER criterion.
            # Open-loop curling past a good grasp destroys it (seed 5:
            # full curl pushes the object to 0.96 m isolation). The
            # pass verdict is force_closure; caging also stops the
            # motion because losing a caged object to keep curling
            # serves no goal.
            if info["caged"] or info["force_closure"]:
                return rewards, info
    _, r, _, info = env.step(np.array(phases[-1][0]))
    rewards.append(r)
    return rewards, info


def run_episode(env, plan, instruction, frames=None, verbose=True):
    """Execute one episode: plan -> singulate -> gate -> cage.

    env must already be reset; target_index is set from the plan here.
    plan: canonical dict from perception.grounding.normalize_plan
    (grasp_target, push_away [{index, effort}], approach_side,
    rationale) — from any planner (VLM, heuristic, scripted).
    frames: list to append renders to (created if None). Returns
    (metrics dict, frames list, cage info).
    Metrics are the comparison surface: force_closure is the goal
    criterion; caged/enclosed/isolation/reward are diagnostics.
    The plan DRIVES execution: a non-empty push_away pushes exactly
    those objects (in order, at their efforts); an empty push_away
    falls back to the executor's crowding filter as a safety net.
    approach_side biases the cage entry yaw. Two different plans on
    the same seed must be able to produce different trajectories —
    otherwise the comparison is vacuous (that was the old bug: the
    plan was used only when the word "push" appeared in the
    instruction, so VLM and heuristic runs were frame-identical).
    """
    if frames is None:
        frames = [env.render()]
    target_idx = plan["grasp_target"]
    env.target_index = target_idx
    # Per-frame grasp status, aligned 1:1 with frames. The initial frame
    # predates any step, so its status comes from _reward() (a pure
    # contact read, no physics step). Singulate/cage append one status
    # per frame they render; reverted pushes truncate both lists.
    _, init_info = env._reward()
    frame_info = [frame_status(init_info)]
    approach_side = plan.get("approach_side", "near")
    if verbose:
        print(f"[demo] instruction: {instruction!r}")
        pushes = ", ".join(
            f"obj{p['index']}({p['effort']})" if isinstance(p, dict)
            else f"obj{p}" for p in plan["push_away"])
        print(f"[demo] plan: grasp object {target_idx} "
              f"({env.objects[target_idx]['color']}), push away "
              f"[{pushes}], approach {approach_side} — "
              f"{plan['rationale']}")

    # The plan drives the pushes: a non-empty push_away pushes exactly
    # those objects (in order, at their efforts). An empty push_away
    # falls back to auto-detecting crowding (the safety net — it does
    # not override a plan that names objects). "without moving" makes
    # the revert strict: any target motion fails the push.
    push_targets = plan["push_away"] if plan["push_away"] else None
    strict = "without moving" in instruction.lower()
    push_log = []
    iso_before, iso_after = scripted_singulate(
        env, target_idx, frames, push_targets=push_targets, strict=strict,
        frame_info=frame_info, push_log=push_log)
    if verbose:
        print(f"[demo] singulation: isolation {iso_before:.3f} -> "
              f"{iso_after:.3f} m")

    # Isolation gate: the task is "grasp IN ISOLATION". Only start the grasp
    # once preconditions are met (or log that they weren't).
    ISOLATION_THRESHOLD = 0.15  # meters of clearance
    preconditions_met = iso_after >= ISOLATION_THRESHOLD
    if verbose:
        if preconditions_met:
            print(f"[demo] preconditions met (isolation {iso_after:.3f} m "
                  f">= {ISOLATION_THRESHOLD} m) — starting grasp")
        else:
            print(f"[demo] WARNING: preconditions NOT met (isolation "
                  f"{iso_after:.3f} m < {ISOLATION_THRESHOLD} m) — "
                  f"attempting grasp anyway")

    rewards, info = scripted_cage(env, target_idx, frames,
                                  frame_info=frame_info,
                                  approach_side=approach_side)
    if verbose:
        print(f"[demo] links touching: {info['links_touching']}, "
              f"enclosed: {info['enclosed']}, caged: {info['caged']}, "
              f"force_closure: {info['force_closure']}, "
              f"total reward: {sum(rewards):.1f}")
        if info["force_closure"]:
            print("[demo] GRASP PASSED (force closure)")
        else:
            print("[demo] grasp FAILED (no force closure)")
    metrics = {
        "target_index": target_idx,
        "target_color": env.objects[target_idx]["color"],
        "rationale": plan["rationale"],
        "push_away": plan["push_away"],
        "approach_side": approach_side,
        "pushes": push_log,
        "isolation_before": iso_before,
        "isolation_after": iso_after,
        "preconditions_met": preconditions_met,
        "links_touching": info["links_touching"],
        "enclosed": info["enclosed"],
        "caged": info["caged"],
        "force_closure": info["force_closure"],
        "total_reward": float(sum(rewards)),
        "frame_info": frame_info,
    }
    return metrics, frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="grasp the red box in isolation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", choices=["vlm", "heuristic"], default="vlm",
                    help="vlm: Nebius VLM (falls back to heuristic on "
                    "failure); heuristic: scripted rule-based planner, "
                    "no API call, no cost")
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

    if args.planner == "heuristic":
        from perception.grounding import HeuristicLabeller
        plan = HeuristicLabeller().plan(frames[0], env.objects,
                                        args.instruction)
    else:
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
    metrics, frames, info = run_episode(env, plan, args.instruction, frames)
    target_idx = metrics["target_index"]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path("episodes") / stamp
    out.mkdir(parents=True, exist_ok=True)
    saved_frames = np.stack(frames)[::args.frame_every]
    np.savez_compressed(out / "frames.npz", frames=saved_frames)
    meta = {
        "instruction": args.instruction,
        "planner": args.planner,
        **metrics,
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
                            fps=args.gif_fps, scale=1.0,
                            frame_info=metrics["frame_info"])
        print(f"[demo] GIF saved to {gif_path}")


if __name__ == "__main__":
    main()

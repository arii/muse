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


def _cage_lead_and_sign(face, target_xy):
    """Lead offset + curl sign for a cage that wraps, not pokes.

    The old cage set q1 = face and always curled CCW (+q2, +q3).
    That fails in two ways, both visible in the compare videos:
    (1) q1 = face points link1 straight through the target's center,
    so the first link hits the box instead of parking beside it;
    (2) sweeping the base toward a clockwise target (-73 deg, seeds
    2/6) while curling CCW smacks the box with the back of link2
    and curls into empty space — the curl must swing back toward
    the target, i.e. opposite the side link1 parks on.
    Rule: delta = wrapped(face - target angle). If |delta| < 0.35 rad
    (near: link1 would point through the box), park link1 0.35 rad
    clockwise of the target and curl CCW (+). Otherwise curl toward
    the target: delta > 0 (link1 CCW of target) -> curl CW (-),
    delta < 0 -> curl CCW (+). Returns (face, curl_sign).
    """
    theta_t = float(np.arctan2(target_xy[1], target_xy[0]))
    delta = (face - theta_t + np.pi) % (2 * np.pi) - np.pi
    if abs(delta) < 0.35:
        return theta_t - 0.35, +1
    return face, (-1 if delta > 0 else +1)


def scripted_cage(env, target_idx, frames, frame_info=None,
                  approach_side="near"):
    """Scripted whole-arm cage around the target object.

    approach_side biases the entry yaw (see approach_face): near is
    the historical behavior (yaw toward the target); far/left/right
    swing the arm in from another side, which matters when a
    distractor sits on the near side. _cage_lead_and_sign then turns
    that yaw into a lead offset + curl sign so the links wrap the
    box instead of poking it.

    Sweep fix (2026-10-28): the old phase 1 yawed the base with the
    arm nearly straight (0.75 m long) — a windshield wiper that
    knocked boxes away before the reach started (seeds 0, 9). The
    new order tucks first, yaws while tucked (swept disc ~0.15 m,
    inside the minimum spawn radius 0.28 m), then extends radially
    with q1 fixed, then reaches and curls.
    Curl is tighter than the old (1.25, 1.45): (1.8, 2.2) brings the
    tip back toward link1, which is what _loop_closed() requires
    for a true cage (old open curl left a 0.32 m tip-link1 gap;
    (1.6, 2.0) still leaves 0.161 m, just over the 0.15 m closure
    threshold, so the final curl must be (1.8, 2.2)).

    Early exit is on force_closure ONLY. The old caged-or-FC exit
    stopped seed 5 mid-reach on a false caged=True (open-arm
    polygon, gap 0.68 m); with the closed-loop metric caged is
    strict, and a true FC grasp is still preserved by the FC exit.
    """
    obs = env._obs()
    pos = obs["objects"][target_idx][:2]
    face = approach_face(pos, approach_side)
    face, sign = _cage_lead_and_sign(face, pos)
    q_cur = env.data.qpos[env._arm_qpos].copy()
    s = float(sign)
    phases = [
        # Tuck tight, not just 1.0 rad: a (1.0, 1.0) fold still leaves
        # the tip ~0.47 m out, inside the target's radius, so yawing
        # still sweeps it. (1.8, 2.0) tucks the tip to ~0.10-0.15 m
        # from the base — inside the minimum spawn radius 0.28 m —
        # so the yaw below cannot touch any spawned box.
        ([q_cur[0], 1.8 * s, 2.0 * s], 60),
        ([face, 1.8 * s, 2.0 * s], 60),
        ([face, 0.15 * s, 0.15 * s], 120),
        ([face, 0.55 * s, 0.55 * s], 120),
        ([face, 1.8 * s, 2.2 * s], 200),
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
            # Stop only on force_closure (the goal criterion).
            # Stopping on caged as well halted seed 5 on a false
            # caged=True; see _loop_closed() and the docstring.
            if info["force_closure"]:
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
    The plan DRIVES execution: push_away pushes exactly those
    objects (in order, at their efforts); an empty push_away means
    push nothing — it does NOT fall back to the crowding filter
    (that fallback, via push_targets=None, is only for direct
    scripted_singulate calls).
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

    # The plan drives the pushes: push_away pushes exactly those
    # objects (in order, at their efforts), and an EMPTY push_away
    # means push nothing. The old code turned [] into None, which
    # activated scripted_singulate's <0.35 m auto-crowding fallback —
    # so a VLM plan that explicitly judged "no preconditions needed"
    # and a heuristic plan that named no push both auto-pushed the
    # same crowding objects with the same "clear" effort, and the
    # 2026-10-28 rerun was identical again. The auto fallback still
    # exists for direct scripted_singulate calls with
    # push_targets=None, but run_episode never uses it: a plan is a
    # decision, including the decision to push nothing. "without
    # moving" makes the revert strict: any target motion fails a push.
    push_targets = plan["push_away"]
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

"""Pipeline verification: exercise every working stage and save screenshots.

Usage:
    MUJOCO_GL=osmesa python scripts/verify_pipeline.py [--seed 4]

Checks:
  1. Grounding  — labeller picks the right object for 3 color instructions.
  2. Segmentation — SAM 3.1 API on a canary frame from a TEXT prompt
     (no GT) when META_API_KEY is set; else local SAM on a GT-derived
     box when SAM_CHECKPOINT is set; else SKIP.
  3. Grasp synthesis — antipodal sampler on the stage-2 mask (M3 live),
     else SKIP with M2. Saves a top-5 grasp overlay figure.
  4. Motion planning — plan() trajectory shape/continuity + executes in env.
  5. Execution — full demo loop (grounding -> singulation -> cage);
     screenshots saved to verification/.

Screenshots land in verification/<stamp>/ as PNGs.
"""
import argparse
import datetime
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from env.whole_arm_grasp import WholeArmGraspEnv
from perception.grounding import make_labeller
from planning.motion import plan
from scripts.demo_vlm_grasp import scripted_cage, scripted_singulate

COLORS = ("red", "blue", "green")


def check_grounding(seed=0):
    print("== [1/5] grounding ==")
    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    rgb = env.render()
    labeller = make_labeller()
    ok = True
    for color in COLORS:
        idx, rationale = labeller.label(
            rgb, env.objects, f"grasp the {color} box")
        got = env.objects[idx]["color"]
        mark = "OK " if got == color else "FAIL"
        if got != color:
            ok = False
        print(f"  [{mark}] 'grasp the {color} box' -> object {idx} ({got})"
              f" — {rationale}")
    return ok


def _red_gt(rgb):
    return ((rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int) > 60)
            & (rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int) > 60))


def check_segmentation(seed=2, outdir=None):
    import os
    print("== [2/5] segmentation ==")
    from perception.segment import text_from_instruction
    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    rgb = env.render()
    h, w = rgb.shape[:2]
    red = _red_gt(rgb)
    assert red.sum() > 50, "canary frame has no red box?"
    via = None
    if os.environ.get("META_API_KEY"):
        from perception.segment import MetaAPISAMSegmenter
        mask = MetaAPISAMSegmenter().segment(
            rgb, {"text": text_from_instruction("grasp the red box")})
        via = "sam-3.1 API (text prompt, no GT)"
    elif os.environ.get("SAM_CHECKPOINT"):
        from perception.segment import SAMSegmenter
        ys, xs = np.nonzero(red)
        box = [max(0, xs.min() - 5), max(0, ys.min() - 5),
               min(w, xs.max() + 5), min(h, ys.max() + 5)]
        mask = SAMSegmenter().segment(rgb, {"boxes": [box]})
        via = f"local SAM (GT-derived box {box})"
    else:
        print("  [SKIP] no META_API_KEY / SAM_CHECKPOINT — M2 unwired.")
        return None, None, None
    assert mask.dtype == bool and mask.shape == (h, w)
    iou = (mask & red).sum() / (mask | red).sum()
    ok = iou > 0.5
    print(f"  [{'OK ' if ok else 'FAIL'}] mask/red IoU {iou:.2f} via {via}")
    if outdir is not None:
        Image.fromarray(rgb).save(outdir / "10_sam_frame.png")
        Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).save(
            outdir / "11_sam_mask.png")
    return ok, mask, rgb


def check_grasp(mask, rgb=None, outdir=None):
    print("== [3/5] grasp synthesis ==")
    if mask is None:
        print("  [SKIP] no stage-2 mask (M2 skipped).")
        return None
    from perception.grasp import synthesize_grasps
    grasps = synthesize_grasps(mask, n_candidates=10)
    ok = len(grasps) > 0
    print(f"  [{'OK ' if ok else 'FAIL'}] {len(grasps)} antipodal "
          f"candidates" + (f", top width {grasps[0]['width']:.1f}px"
                            if ok else ""))
    if ok and rgb is not None and outdir is not None:
        from PIL import ImageDraw
        fig = Image.fromarray(np.asarray(rgb)).convert("RGB")
        d = ImageDraw.Draw(fig)
        for g in grasps[:5]:
            x, y, a, wdt = g["x"], g["y"], g["angle"], g["width"] / 2
            dx, dy = np.cos(a) * wdt, np.sin(a) * wdt
            d.line([(x - dx, y - dy), (x + dx, y + dy)],
                   fill=(0, 255, 0), width=2)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 215, 0))
        fig.save(outdir / "12_grasp_candidates.png")
        print(f"  figure -> {outdir / '12_grasp_candidates.png'}")
    return ok


def check_motion(seed=0):
    print("== [4/5] motion planning ==")
    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    q0 = env._obs()["qpos"]
    q_goal = np.array([0.5, 0.9, 1.1])
    traj = plan(q0, q_goal, n_steps=60)
    assert traj.shape == (60, 3), traj.shape
    assert np.allclose(traj[0], q0) and np.allclose(traj[-1], q_goal)
    max_step = np.abs(np.diff(traj, axis=0)).max()
    print(f"  [OK ] trajectory shape {traj.shape}, endpoints exact, "
          f"max per-step joint delta {max_step:.4f} rad")
    for q in traj[::15]:
        env.step(q)
    print("  [OK ] waypoints execute in env without error")
    return True


def run_demo_screenshots(seed, outdir):
    print("== [5/5] execution (demo loop + screenshots) ==")
    env = WholeArmGraspEnv(seed=seed)
    env.reset(seed=seed)
    frames = [env.render()]
    labeller = make_labeller()
    target_idx, rationale = labeller.label(
        frames[0], env.objects, "grasp the red box")
    env.target_index = target_idx
    print(f"  instruction 'grasp the red box' -> object {target_idx} "
          f"({env.objects[target_idx]['color']})")
    iso_before, iso_after = scripted_singulate(env, target_idx, frames)
    post_singulate = frames[-1]
    print(f"  singulation: isolation {iso_before:.3f} -> {iso_after:.3f} m")
    rewards, info = scripted_cage(env, target_idx, frames)
    frames.append(env.render())
    print(f"  links touching: {info['links_touching']}, "
          f"caged: {info['caged']}, total reward: {sum(rewards):.1f}, "
          f"frames: {len(frames)}")

    n = len(frames)
    picks = {
        "00_initial": frames[0],
        "01_singulated": post_singulate,
        "02_caging": frames[2 * n // 3],
        "03_final": frames[-1],
    }
    for name, fr in picks.items():
        p = outdir / f"{name}.png"
        Image.fromarray(fr).save(p)
        print(f"  screenshot -> {p}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path("verification") / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    g = check_grounding(seed=args.seed)
    s, mask, rgb = check_segmentation(outdir=outdir)
    gs = check_grasp(mask, rgb, outdir)
    m = check_motion(seed=args.seed)
    info = run_demo_screenshots(args.seed, outdir)

    print("\n== summary ==")
    print(f"  grounding: {'PASS' if g else 'FAIL'}")
    print(f"  segmentation: {'PASS' if s else ('SKIP' if s is None else 'FAIL')}")
    print(f"  grasp synthesis: {'PASS' if gs else ('SKIP' if gs is None else 'FAIL')}")
    print(f"  motion planning: {'PASS' if m else 'FAIL'}")
    print(f"  execution: {'CAGED' if info['caged'] else 'no cage'} "
          f"(seed {args.seed})")
    print(f"  screenshots: {outdir}")


if __name__ == "__main__":
    main()

"""Pipeline tests — headless (MUJOCO_GL=osmesa).

Run:  MUJOCO_GL=osmesa pytest -v
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from env.whole_arm_grasp import WholeArmGraspEnv  # noqa: E402
from perception.grounding import make_labeller  # noqa: E402
from planning.motion import plan  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_grounding_picks_correct_color():
    """Instruction -> labeller chooses the object of the named color."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    rgb = env.render()
    labeller = make_labeller()
    for color in ("red", "blue", "green"):
        idx, _ = labeller.label(rgb, env.objects, f"grasp the {color} box")
        assert env.objects[idx]["color"] == color, \
            f"instruction said {color}, labeller chose {env.objects[idx]['color']}"


def test_motion_plan_endpoints_and_continuity():
    """plan() returns a (T, 3) trajectory from q_start to q_goal."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    q0 = env._obs()["qpos"]
    q_goal = np.array([0.5, 0.9, 1.1])
    traj = plan(q0, q_goal, n_steps=60)
    assert traj.shape == (60, 3)
    np.testing.assert_allclose(traj[0], q0)
    np.testing.assert_allclose(traj[-1], q_goal)
    assert np.abs(np.diff(traj, axis=0)).max() < 0.1
    for q in traj[::15]:  # waypoints execute without error
        env.step(np.asarray(q))


def test_env_renders_headless():
    """Top-down render works without a display and isn't blank."""
    env = WholeArmGraspEnv(seed=1)
    env.reset(seed=1)
    rgb = env.render()
    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8
    assert rgb.std() > 1.0


def test_demo_end_to_end(tmp_path):
    """scripts/demo_vlm_grasp.py runs and logs an episode (frames + meta)."""
    proc = subprocess.run(
        [sys.executable, "scripts/demo_vlm_grasp.py",
         "--instruction", "grasp the red box", "--seed", "4"],
        cwd=REPO,
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "MUJOCO_GL": "osmesa",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "episode saved to" in proc.stdout
    epi = sorted((REPO / "episodes").glob("*"))[-1]
    meta = json.loads((epi / "meta.json").read_text())
    assert meta["instruction"] == "grasp the red box"
    assert meta["target_color"] == "red"
    assert (epi / "frames.npz").exists()
    frames = np.load(epi / "frames.npz")["frames"]
    assert frames.ndim == 4 and frames.shape[1:] == (480, 640, 3)

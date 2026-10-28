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
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "osmesa")

from env.whole_arm_grasp import WholeArmGraspEnv  # noqa: E402
from perception.grounding import make_labeller  # noqa: E402
from planning.motion import plan  # noqa: E402
from scripts.make_video import make_gif  # noqa: E402

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


def test_heuristic_plan_parses_push_instruction():
    """'push the blue ... away' -> push_away names the blue object."""
    from perception.grounding import HeuristicLabeller
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab = HeuristicLabeller()
    plan = lab.plan(env.render(), env.objects,
                    "grasp the red box, first push the blue box away "
                    "without moving the red box")
    assert plan["grasp_target"] == 0  # red
    assert plan["push_away"] == [1]  # blue
    # plain grasp instruction: no push_away, grasp still parsed
    plan2 = lab.plan(env.render(), env.objects, "grasp the green box")
    assert plan2["grasp_target"] == 2 and plan2["push_away"] == []


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


def test_make_video_writes_gif(tmp_path):
    """make_gif renders an animated GIF with one frame per input frame."""
    frames = np.random.randint(0, 255, (5, 48, 64, 3), dtype=np.uint8)
    meta = {"instruction": "grasp the red box",
            "target_color": "red", "caged": True}
    out = tmp_path / "demo.gif"
    make_gif(frames, meta, out, fps=8, scale=1.0)
    assert out.exists()
    with Image.open(out) as img:
        assert img.n_frames == 5


# --- NebiusVLMLabeller, offline (HTTP layer mocked) ---

def _mock_labeller(monkeypatch, reply_text):
    """NebiusVLMLabeller with a canned HTTP response; returns (labeller, sent)."""
    from perception.grounding import NebiusVLMLabeller
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    sent = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": reply_text}}]}

    class FakeRequests:
        def post(self, url, headers=None, json=None, timeout=None):
            sent.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResp()

    lab = NebiusVLMLabeller()
    lab._requests = FakeRequests()
    return lab, sent


def test_nebius_labeller_request_shape_and_parse(monkeypatch):
    """The VLM request carries text+image and the noisy JSON reply parses."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, sent = _mock_labeller(
        monkeypatch,
        'Sure! {"grasp_target": 0, "push_away": [1], '
        '"rationale": "the red box"} done.')

    plan = lab.label(env.render(), env.objects, "grasp the red box")
    # label() stays back-compat: (grasp_target, rationale)
    assert plan == (0, "the red box")
    assert lab.last_raw.startswith("Sure!")
    assert sent["url"] == \
        "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    body = sent["json"]
    assert body["model"] == "nvidia/Nemotron-3.5-Super-VL"
    assert body["temperature"] == 0
    content = body["messages"][0]["content"]
    assert {c["type"] for c in content} == {"text", "image_url"}
    text = next(c["text"] for c in content if c["type"] == "text")
    assert "grasp the red box" in text
    assert "index 0: red box" in text
    assert "push_away" in text
    img_url = next(c["image_url"]["url"] for c in content
                   if c["type"] == "image_url")
    assert img_url.startswith("data:image/png;base64,")


def test_nebius_labeller_plan_parses_push_away(monkeypatch):
    """plan() returns grasp_target + push_away list; rejects bad indices."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, _ = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [1, 2], "rationale": "clear it"}')
    plan = lab.plan(env.render(), env.objects, "grasp the red box")
    assert plan == {"grasp_target": 0, "push_away": [1, 2],
                    "rationale": "clear it"}

    lab2, _ = _mock_labeller(monkeypatch, '{"grasp_target": 9, "push_away": []}')
    with pytest.raises(ValueError):
        lab2.plan(env.render(), env.objects, "grasp the red box")


def test_nebius_labeller_rejects_non_json(monkeypatch):
    """A reply with no JSON object raises instead of guessing."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, _ = _mock_labeller(monkeypatch, "I cannot see any image.")
    with pytest.raises(ValueError):
        lab.label(env.render(), env.objects, "grasp the red box")


def test_nebius_labeller_null_content_diagnostic(monkeypatch):
    """content=None raises a diagnostic (not AttributeError), and
    reasoning_content is used as a fallback."""
    from perception.grounding import NebiusVLMLabeller
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    class FakeRespNull:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": None,
                                             "role": "assistant"}}]}

    class FakeRequestsNull:
        def post(self, url, headers=None, json=None, timeout=None):
            return FakeRespNull()

    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab = NebiusVLMLabeller()
    lab._requests = FakeRequestsNull()
    with pytest.raises(ValueError) as exc:
        lab.plan(env.render(), env.objects, "grasp the red box")
    # diagnostic names the model and the likely cause, not just "None"
    assert "nvidia/Nemotron-3.5-Super-VL" in str(exc.value)
    assert "vision" in str(exc.value).lower()

    # reasoning_content fallback works
    class FakeRespReasoning:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {
                "content": None,
                "reasoning_content": '{"grasp_target": 0, "push_away": []}',
                "role": "assistant"}}]}

    class FakeRequestsReasoning:
        def post(self, url, headers=None, json=None, timeout=None):
            return FakeRespReasoning()

    lab2 = NebiusVLMLabeller()
    lab2._requests = FakeRequestsReasoning()
    plan = lab2.plan(env.render(), env.objects, "grasp the red box")
    assert plan["grasp_target"] == 0


def test_singulation_improves_isolation():
    """scripted_singulate pushes a crowding object away (seed 7)."""
    from scripts.demo_vlm_grasp import scripted_singulate
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    env.target_index = 0
    assert env.target_isolation() < 0.15  # seed 7 starts crowded
    iso_before, iso_after = scripted_singulate(env, 0, frames=[])
    assert iso_after > iso_before + 0.05, \
        f"isolation did not improve: {iso_before:.3f} -> {iso_after:.3f}"


def test_singulation_never_regresses():
    """Across seeds, singulation never reduces isolation (revert on harm)."""
    from scripts.demo_vlm_grasp import scripted_singulate
    for seed in range(6):
        env = WholeArmGraspEnv(seed=seed)
        env.reset(seed=seed)
        env.target_index = 0
        iso_before, iso_after = scripted_singulate(env, 0, frames=[])
        assert iso_after >= iso_before - 1e-9, \
            f"seed {seed}: isolation regressed {iso_before:.3f} -> {iso_after:.3f}"


def test_ik_reaches_targets():
    """DLS IK solves to <1mm for reachable targets."""
    import numpy as np
    from planning.primitives import fk, ik_dls
    rng = np.random.default_rng(0)
    for _ in range(10):
        r = rng.uniform(0.2, 0.7)
        th = rng.uniform(-2.5, 2.5)
        target = np.array([r * np.cos(th), r * np.sin(th)])
        q = ik_dls(target, np.zeros(3))
        assert np.linalg.norm(fk(q) - target) < 1e-3


def test_push_moves_object_along_theta():
    """push() displaces the object in the push direction."""
    import numpy as np
    from planning.primitives import push
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    p0 = env._obs()["objects"][0][:2].copy()
    sx = env.objects[0]["half"][0]
    theta = 0.0  # +x
    direction = np.array([np.cos(theta), np.sin(theta)])
    # start clearly outside: box half + tip radius + margin
    start = p0 - direction * (sx + 0.025 + 0.015)
    push(env, start, theta, 0.15, frames=[])
    moved = env._obs()["objects"][0][:2] - p0
    # primitive moves the box in the push direction (exact distance varies
    # with contact dynamics; singulate's revert logic handles the rest)
    assert moved[0] > 0.015, f"push did not move +x: {moved}"

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
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": reply_text}}]}

    class FakeRequests:
        def post(self, url, headers=None, json=None, timeout=None):
            sent.update(url=url, headers=headers, json=json, timeout=timeout)
            return FakeResp()

    lab = NebiusVLMLabeller(ledger=False)
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
    """content=None raises a diagnostic (not AttributeError)."""
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
    lab = NebiusVLMLabeller(ledger=False)
    lab._requests = FakeRequestsNull()
    with pytest.raises(ValueError) as exc:
        lab.plan(env.render(), env.objects, "grasp the red box")
    # diagnostic names the model and the likely cause, not just "None"
    assert "nvidia/Nemotron-3.5-Super-VL" in str(exc.value)
    assert "vision" in str(exc.value).lower()


def test_nebius_labeller_sends_response_format(monkeypatch):
    """plan() sends response_format json_schema (structured output)."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, sent = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [1], "rationale": "clear it"}')
    plan = lab.plan(env.render(), env.objects, "grasp the red box")
    assert plan["grasp_target"] == 0
    rf = sent["json"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert set(schema["required"]) == {"grasp_target", "push_away",
                                       "rationale"}
    assert schema["properties"]["grasp_target"]["type"] == "integer"


def test_vlm_ledger_caches_identical_calls(monkeypatch, tmp_path):
    """Second identical plan() reuses the ledger — no second API call."""
    import json
    from perception.grounding import NebiusVLMLabeller
    from perception.ledger import VLMCallLedger
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    calls = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content":
                '{"grasp_target": 0, "push_away": [1], '
                '"rationale": "clear it"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                          "total_tokens": 120}}

    class FakeRequests:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(json)
            return FakeResp()

    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    rgb = env.render()
    ledger = VLMCallLedger(tmp_path / "ledger.jsonl")
    lab = NebiusVLMLabeller(ledger=ledger)
    lab._requests = FakeRequests()

    plan1 = lab.plan(rgb, env.objects, "grasp the red box")
    plan2 = lab.plan(rgb, env.objects, "grasp the red box")
    assert plan1 == plan2 == {"grasp_target": 0, "push_away": [1],
                              "rationale": "clear it"}
    assert len(calls) == 1, "second identical call must not hit the API"

    entries = [json.loads(l) for l in
               (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(entries) == 2
    assert entries[0]["cache_hit"] is False
    assert entries[0]["usage"]["total_tokens"] == 120
    assert entries[0]["plan"] == plan1
    assert entries[1]["cache_hit"] is True
    assert entries[1]["plan"] == plan1
    assert entries[1]["cache_key"] == entries[0]["cache_key"]

    # different instruction misses the cache and calls the API again
    lab.plan(rgb, env.objects, "grasp the blue box")
    assert len(calls) == 2


def test_vlm_ledger_records_failures_without_caching(monkeypatch, tmp_path):
    """A failed call is logged (with tokens) but never served from cache."""
    import json
    from perception.grounding import NebiusVLMLabeller
    from perception.ledger import VLMCallLedger
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    class FakeRespFail:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": None,
                                             "reasoning_content": "...",
                                             "role": "assistant"}}],
                    "usage": {"prompt_tokens": 519, "completion_tokens": 800,
                              "total_tokens": 1319}}

    class FakeReqFail:
        def post(self, url, headers=None, json=None, timeout=None):
            return FakeRespFail()

    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    rgb = env.render()
    ledger = VLMCallLedger(tmp_path / "ledger.jsonl")
    lab = NebiusVLMLabeller(ledger=ledger)
    lab._requests = FakeReqFail()
    with pytest.raises(ValueError):
        lab.plan(rgb, env.objects, "grasp the red box")
    entries = [json.loads(l) for l in
               (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["plan"] is None
    assert entries[0]["usage"]["total_tokens"] == 1319
    assert "error" in entries[0]

    # retry with a working API still calls it (failure not cached)
    calls = []

    class FakeRespOK:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content":
                '{"grasp_target": 0, "push_away": [], "rationale": "ok"}'}}],
                "usage": {}}

    class FakeReqOK:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(1)
            return FakeRespOK()

    lab._requests = FakeReqOK()
    plan = lab.plan(rgb, env.objects, "grasp the red box")
    assert plan["grasp_target"] == 0
    assert len(calls) == 1


def test_force_closure_math():
    """_force_closure: positive span of wrenches, no scipy needed."""
    import math
    import numpy as np
    fc = WholeArmGraspEnv._force_closure
    # tetrahedron around origin -> True
    tet = [np.array([1, 1, 1.]), np.array([1, -1, -1.]),
           np.array([-1, 1, -1.]), np.array([-1, -1, 1.])]
    assert fc(tet) is True
    # all wrenches in a half-space -> False
    half = [np.array([-1, 0.2, 0.1]), np.array([-1, -0.3, 0.2]),
            np.array([-0.5, 1, 0.]), np.array([-0.5, -1, 0.])]
    assert fc(half) is False
    # fewer than 4 wrenches cannot span R^3 -> False
    assert fc(tet[:3]) is False
    # antipodal pair (normals along the segment, friction cones
    # overlapping) -> True; same-side pair -> False
    mu = 0.8
    a = math.atan(mu)

    def cone(n, c):
        t = np.array([-n[1], n[0]])
        out = []
        for s in (+1, -1):
            f = n * math.cos(a) + t * (s * math.sin(a))
            out.append(np.array([f[0], f[1], c[0] * f[1] - c[1] * f[0]]))
        return out

    anti = (cone(np.array([0., -1.]), np.array([0., 0.05])) +
            cone(np.array([0., 1.]), np.array([0., -0.05])))
    assert fc(anti) is True
    same = (cone(np.array([0., -1.]), np.array([-0.05, 0.])) +
            cone(np.array([0., -1.]), np.array([0.05, 0.])))
    assert fc(same) is False


def test_force_closure_vs_caging_seed5():
    """Seed 5 is geometrically caged but NOT force closure.

    Caging and force closure are distinct: the arm wraps the box
    (enclosed, 2 links) but all contacts sit on adjacent sides, so
    the wrenches cannot positively span R^3. Documents that the
    enveloping cage does not yet achieve a true grasp. The reward
    bonus follows force_closure (the goal criterion), so the final
    step earns 0.5*2 = 1.0 with no +2.0 bonus.
    """
    from scripts.demo_vlm_grasp import scripted_cage
    env = WholeArmGraspEnv(seed=5)
    env.reset(seed=5)
    env.target_index = 0
    rewards, info = scripted_cage(env, 0, frames=[])
    assert info["caged"] is True
    assert info["force_closure"] is False
    assert rewards[-1] == 1.0


def test_run_episode_heuristic_no_api(monkeypatch):
    """run_episode with a heuristic plan returns full metrics, no API."""
    from scripts.demo_vlm_grasp import run_episode
    from perception.grounding import HeuristicLabeller
    env = WholeArmGraspEnv(seed=5)
    env.reset(seed=5)
    frames = [env.render()]
    plan = HeuristicLabeller().plan(frames[0], env.objects,
                                    "grasp the red box in isolation")
    metrics, _, info = run_episode(env, plan,
                                   "grasp the red box in isolation",
                                   frames, verbose=False)
    assert metrics["target_index"] == 0
    assert metrics["target_color"] == "red"
    assert metrics["caged"] is True
    assert metrics["force_closure"] is False
    assert metrics["total_reward"] == 1.0  # 0.5*2, no FC bonus
    assert info["force_closure"] is False


def test_compare_aggregate():
    """compare_planners.aggregate computes rates and means correctly."""
    from scripts.compare_planners import aggregate
    rows = [
        {"planner": "heuristic", "planner_error": None,
         "force_closure": True, "caged": True,
         "preconditions_met": True, "total_reward": 3.0,
         "isolation_before": 0.1, "isolation_after": 0.2,
         "api_call": False, "tokens": 0},
        {"planner": "heuristic", "planner_error": None,
         "force_closure": False, "caged": False,
         "preconditions_met": True, "total_reward": 1.0,
         "isolation_before": 0.1, "isolation_after": 0.15,
         "api_call": False, "tokens": 0},
        {"planner": "heuristic", "planner_error": "boom",
         "api_call": False, "tokens": 0},
    ]
    agg = aggregate(rows, "heuristic")
    assert agg["episodes"] == 3
    assert agg["planner_errors"] == 1
    assert agg["force_closure_rate"] == 0.5
    assert agg["caged_rate"] == 0.5
    assert agg["mean_reward"] == 2.0
    assert agg["api_calls"] == 0
    assert agg["total_tokens"] == 0


def test_make_side_by_side_gif(tmp_path):
    """Side-by-side GIF: header/footer, padding to the longer sequence."""
    import numpy as np
    from PIL import Image
    from scripts.make_video import make_side_by_side_gif
    # Left: 3 red frames; right: 5 blue frames (longer -> left pads).
    # Vary one pixel per frame: PIL collapses byte-identical GIF frames,
    # so distinct frames are needed to count them.
    left = []
    for i in range(3):
        fr = np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8)
        fr[0, 0] = (i, 0, 0)
        left.append(fr)
    right = []
    for i in range(5):
        fr = np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8)
        fr[0, 0] = (0, 0, i)
        right.append(fr)
    out = tmp_path / "sbs.gif"
    make_side_by_side_gif(left, right, "vlm", "heuristic",
                          "FC=False | reward=1.5",
                          "FC=False | reward=1.5",
                          out, fps=15, scale=1.0)
    assert out.exists() and out.stat().st_size > 0
    im = Image.open(out)
    assert im.n_frames == 5  # padded to the longer sequence
    # Width: 60 + 4 divider + 60; height: 28 header + 40 + 28 footer.
    assert im.size == (124, 96)


def test_compare_run_one_returns_frames():
    """run_one returns (row, frames) for the compare GIF path."""
    from scripts.compare_planners import run_one
    # Seed 7: long episode (singulate push + full cage, no early stop).
    row, frames = run_one(7, "heuristic", "grasp the red box in isolation")
    assert row["planner"] == "heuristic"
    assert row["planner_error"] is None
    assert len(frames) > 50
    assert frames[0].shape == (480, 640, 3)


def test_cage_requires_enclosure_not_just_contact():
    """Two links on the same side is a poke, not a cage.

    Straight arm, box straddling the link1-link2 joint so it touches
    both links. Old metric (n>=2) called this caged; the enclosure
    test must reject it — the box is outside the degenerate polygon.
    """
    import mujoco
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    env.target_index = 0
    env.data.qpos[env._arm_qpos] = [0, 0, 0]  # straight along +x
    adr = env._obj_qposadr[0]
    env.data.qpos[adr:adr + 3] = [0.30, 0.05, 0.06]
    mujoco.mj_forward(env.model, env.data)
    _, info = env._reward()
    assert info["links_touching"] == 2, \
        f"test setup broken: expected 2 links touching, got {info}"
    assert info["enclosed"] is False
    assert info["caged"] is False


def test_scripted_cage_achieves_true_enclosure():
    """The scripted cage still counts as caged under the enclosure test.

    Seed 5: the curl wraps the target with 2 links touching and the
    target center inside the arm polygon. (The scripted cage is
    brittle — most seeds miss entirely — but when it wraps, the new
    metric must credit it.)
    """
    from scripts.demo_vlm_grasp import scripted_cage
    env = WholeArmGraspEnv(seed=5)
    env.reset(seed=5)
    env.target_index = 0
    _, info = scripted_cage(env, 0, frames=[])
    assert info["caged"] is True, f"scripted cage failed: {info}"
    assert info["enclosed"] is True
    assert info["links_touching"] >= 2


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

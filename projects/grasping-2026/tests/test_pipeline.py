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
    assert plan["push_away"] == [{"index": 1, "effort": "clear"}]  # blue
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
    # Parse the episode dir from the demo's own output — globbing
    # episodes/ is fragile (compare runs leave compare-*.html/dirs
    # there that sort after the date-stamped episode dirs).
    epi = next(REPO / line.rsplit("episode saved to ", 1)[1].strip()
               for line in proc.stdout.splitlines()
               if "episode saved to" in line)
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
    """plan() returns the canonical shape; rejects bad indices/effort."""
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, _ = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [1, 2], "rationale": "clear it"}')
    plan = lab.plan(env.render(), env.objects, "grasp the red box")
    # Legacy bare indices normalize to effort "clear", missing side
    # defaults to "near" — old replies and cached plans keep working.
    assert plan == {"grasp_target": 0,
                    "push_away": [{"index": 1, "effort": "clear"},
                                  {"index": 2, "effort": "clear"}],
                    "approach_side": "near",
                    "rationale": "clear it"}

    lab_new, _ = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [{"index": 2, "effort": '
        '"shove"}], "approach_side": "far", "rationale": "jammed"}')
    plan_new = lab_new.plan(env.render(), env.objects, "grasp the red box")
    assert plan_new["push_away"] == [{"index": 2, "effort": "shove"}]
    assert plan_new["approach_side"] == "far"

    lab2, _ = _mock_labeller(monkeypatch, '{"grasp_target": 9, "push_away": []}')
    with pytest.raises(ValueError):
        lab2.plan(env.render(), env.objects, "grasp the red box")

    lab3, _ = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [{"index": 1, "effort": '
        '"yeet"}], "approach_side": "near", "rationale": "x"}')
    with pytest.raises(ValueError):
        lab3.plan(env.render(), env.objects, "grasp the red box")


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
                                       "approach_side", "rationale"}
    assert schema["properties"]["grasp_target"]["type"] == "integer"


def test_vlm_prompt_no_anchor_example_and_brief_reasoning(monkeypatch):
    """Prompt has no concrete-index example and no conflicting ban.

    Regression for the 2026-10-28 failure-mode review: the old prompt
    said 'no explanation, no reasoning steps' while demanding a
    rationale field, and gave a concrete '{"grasp_target": 2, ...}'
    example for a red-box (index 0) task — an anchor bias for smaller
    VLMs. The new prompt puts explanation strictly inside rationale,
    asks for brief internal reasoning, and gives no concrete example
    (the strict schema defines the shape).
    """
    from perception.grounding import PROMPT
    assert '"grasp_target": 2' not in PROMPT
    assert "no reasoning steps" not in PROMPT
    assert "strictly inside the 'rationale' field" in PROMPT
    assert "Keep internal reasoning brief" in PROMPT
    # approach_side anchored, not vague: base + image CCW definition
    assert "side facing the base" in PROMPT
    assert "counterclockwise" in PROMPT
    assert "+x toward +y" in PROMPT


def test_vlm_request_reasoning_effort_and_max_tokens(monkeypatch):
    """plan() sends low reasoning_effort and an 8192 token cap.

    Regression for the 2026-10-28 S5 rerun: content null at the 4096
    cap (07:44 UTC) because Nemotron spent the whole budget thinking.
    """
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab, sent = _mock_labeller(
        monkeypatch,
        '{"grasp_target": 0, "push_away": [], "approach_side": "near", '
        '"rationale": "isolated"}')
    lab.plan(env.render(), env.objects, "grasp the red box")
    assert sent["json"]["reasoning_effort"] == "low"
    assert sent["json"]["max_tokens"] == 8192


def test_vlm_null_content_names_reasoning_exhaustion(monkeypatch):
    """finish_reason=length + reasoning==completion names the cause."""
    from perception.grounding import NebiusVLMLabeller
    monkeypatch.setenv("NEBIUS_VLM_MODEL", "nvidia/Nemotron-3.5-Super-VL")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")

    class FakeRespExhaust:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"finish_reason": "length",
                                 "message": {"content": None,
                                             "role": "assistant"}}],
                    "usage": {"completion_tokens": 4096,
                              "completion_tokens_details":
                                  {"reasoning_tokens": 4096}}}

    class FakeRequestsExhaust:
        def post(self, url, headers=None, json=None, timeout=None):
            return FakeRespExhaust()

    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    lab = NebiusVLMLabeller(ledger=False)
    lab._requests = FakeRequestsExhaust()
    with pytest.raises(ValueError) as exc:
        lab.plan(env.render(), env.objects, "grasp the red box")
    assert "REASONING TOKEN EXHAUSTION" in str(exc.value)
    assert "NEBIUS_REASONING_EFFORT" in str(exc.value)


def test_spawn_in_enveloping_workspace():
    """Every spawned box is in the arm's enveloping workspace.

    Regression for the compare-report review seeds 0-9: the old
    r in [0.28, 0.55], theta in [-pi, pi] spawned boxes j1 cannot
    reach (|theta| > 2.6) and boxes link1 cannot touch (r > 0.40).
    """
    for seed in range(20):
        env = WholeArmGraspEnv(seed=seed)
        env.reset(seed=seed)
        for o in env.objects:
            x, y = o["spawn_xy"]
            r = (x * x + y * y) ** 0.5
            import math
            th = math.atan2(y, x)
            assert 0.28 <= r <= 0.40, (seed, o, r)
            assert abs(th) <= 2.4, (seed, o, th)


def test_cage_lead_and_sign():
    """Lead offset parks link1 beside the box; curl swings back.

    Near (face == target angle) -> face shifted -0.35 rad, curl +.
    Left (face +90 deg) -> curl -, right (face -90 deg) -> curl +.
    """
    from scripts.demo_vlm_grasp import _cage_lead_and_sign
    import math
    face, sign = _cage_lead_and_sign(0.5, (math.cos(0.5), math.sin(0.5)))
    assert sign == +1
    assert abs(face - (0.5 - 0.35)) < 1e-9
    _, sign_l = _cage_lead_and_sign(0.5 + math.pi / 2,
                                    (math.cos(0.5), math.sin(0.5)))
    assert sign_l == -1
    _, sign_r = _cage_lead_and_sign(0.5 - math.pi / 2,
                                    (math.cos(0.5), math.sin(0.5)))
    assert sign_r == +1


def test_loop_closed_requires_tight_curl():
    """Open-arm polygon is not a cage; tight curl closes the loop.

    Seed 5 phase 2 (0.55, 0.55) left a 0.68 m tip-base gap and was
    counted caged; a tight curl (1.8, 2.2) brings the tip within
    0.15 m of link1. Straight/open poses must report not closed.
    (1.6, 2.0) still leaves 0.161 m, just over the threshold —
    that is why the cage's final curl is (1.8, 2.2), not (1.6, 2.0).
    """
    import mujoco
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    env.target_index = 0
    env.data.qpos[env._arm_qpos] = [0, 0.55, 0.55]
    mujoco.mj_forward(env.model, env.data)
    assert env._loop_closed() is False
    env.data.qpos[env._arm_qpos] = [0, 1.8, 2.2]
    mujoco.mj_forward(env.model, env.data)
    assert env._loop_closed() is True


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
    assert plan1 == plan2 == {"grasp_target": 0,
                              "push_away": [{"index": 1,
                                             "effort": "clear"}],
                              "approach_side": "near",
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
    """Caging is distinct from force closure, and seed 5 is no
    longer a false caged=True.

    Old seed 5 (old spawn + old polygon metric) was counted caged
    on an open arm. Under the new spawn and the closed-loop metric,
    the scripted cage on seed 5 claims neither caged nor force
    closure — the honest result for a cage that ends not touching
    the target. The positive caging-vs-FC case is covered
    synthetically in test_scripted_cage_achieves_true_enclosure.
    """
    from scripts.demo_vlm_grasp import scripted_cage
    env = WholeArmGraspEnv(seed=5)
    env.reset(seed=5)
    env.target_index = 0
    _, info = scripted_cage(env, 0, frames=[])
    assert info["force_closure"] is False
    assert info["caged"] is False


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
    assert metrics["force_closure"] is False
    assert info["force_closure"] is False
    # No API: heuristic plan, and no false caged claim under the
    # strict closed-loop metric (old test asserted caged True on
    # the seed-5 false positive).


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
    # Move a white dot per frame: PIL collapses byte-identical GIF
    # frames, so distinct frames are needed to count them. The dot
    # must be a color already in the first frame's palette (white,
    # from the header text) — a 1-pixel (i, 0, 0) change quantizes
    # to black under the shared-palette GIF save and collapses.
    left = []
    for i in range(3):
        fr = np.full((40, 60, 3), (255, 0, 0), dtype=np.uint8)
        fr[i + 1, i + 1] = (255, 255, 255)
        left.append(fr)
    right = []
    for i in range(5):
        fr = np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8)
        fr[i + 1, i + 1] = (255, 255, 255)
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
    # Planar joints: place via spawn-relative slides, not raw qpos
    # (qpos is now [slide_x, slide_y, yaw], not [x, y, z, quat]).
    env.set_object_pose(0, [0.30, 0.05], yaw=0.0)
    _, info = env._reward()
    assert info["links_touching"] == 2, \
        f"test setup broken: expected 2 links touching, got {info}"
    assert info["enclosed"] is False
    assert info["caged"] is False


def test_scripted_cage_achieves_true_enclosure():
    """A truly closed wrap counts as caged, and still is not FC.

    Synthetic, not seed-locked: tight curl (1.8, 2.2) with the box
    placed inside the loop at (0.25, 0.20) touches 2 links, is
    enclosed, and the loop is closed — so caged True — while the
    contacts sit on adjacent sides, so force_closure stays False.
    This is the caging-vs-FC distinction the old seed-5 test
    tried to lock in, without relying on that false positive.
    """
    import mujoco
    env = WholeArmGraspEnv(seed=0)
    env.reset(seed=0)
    env.target_index = 0
    env.data.qpos[env._arm_qpos] = [0, 1.8, 2.2]
    mujoco.mj_forward(env.model, env.data)
    env.set_object_pose(0, [0.25, 0.20], yaw=0.0)
    _, info = env._reward()
    assert info["caged"] is True, f"closed wrap not caged: {info}"
    assert info["enclosed"] is True
    assert info["links_touching"] >= 2
    assert info["force_closure"] is False


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
    """push() displaces the object in the push direction.

    Geometry matters (2026-10-28): the old seed-7 target sits at
    distance 0.448 from the base, so a tip push from its west face
    needs a wrist at distance 0.706 — outside the 0.55 m reach of
    the first two links. A tip push there is geometrically
    impossible without link3 crossing the box, and the old test
    only passed because the freejoint box tipped/rolled east.
    This test uses a feasible geometry instead: target at
    (0.45, 0), others moved away, oriented link3 (see
    ik_tip_oriented) — push +x must move the box +x. Exact
    distance varies with contact dynamics and approach sweep;
    singulate's revert logic handles the rest.
    """
    import numpy as np
    from planning.primitives import push
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    for idx, xy in [(1, [-0.5, -0.5]), (2, [0.5, 0.5]),
                    (3, [-0.5, 0.5])]:
        env.set_object_pose(idx, xy)
    env.set_object_pose(0, [0.45, 0.0])
    p0 = env._obs()["objects"][0][:2].copy()
    sx = env.objects[0]["half"][0]
    theta = 0.0  # +x
    direction = np.array([np.cos(theta), np.sin(theta)])
    # start clearly outside: box half + tip radius + margin
    start = p0 - direction * (sx + 0.025 + 0.015)
    push(env, start, theta, 0.15, frames=[])
    moved = env._obs()["objects"][0][:2] - p0
    assert moved[0] > 0.015, f"push did not move +x: {moved}"


def test_gym_wrapper_spaces_and_api():
    """Gymnasium wrapper: 5-tuple step, correct spaces, target=red."""
    from env.gym_wrapper import GraspGymWrapper
    env = GraspGymWrapper(seed=0, max_steps=50)
    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert obs.shape == (16,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    assert action.shape == (3,)
    assert env.action_space.low.tolist() == [-1.0] * 3
    assert env.action_space.high.tolist() == [1.0] * 3
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "force_closure" in info2 and "caged" in info2


def test_gym_wrapper_truncates_on_max_steps():
    """Truncation fires at max_steps without a force-closure success."""
    from env.gym_wrapper import GraspGymWrapper
    env = GraspGymWrapper(seed=0, max_steps=3)
    env.reset(seed=0)
    for i in range(3):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        if i < 2:
            assert not truncated
        else:
            assert truncated and not terminated


def test_gym_wrapper_passes_sb3_check_env():
    """SB3's env checker validates the wrapper (spaces, reset, step)."""
    from stable_baselines3.common.env_checker import check_env
    from env.gym_wrapper import GraspGymWrapper
    check_env(GraspGymWrapper(seed=0, max_steps=10), warn=True)


def test_gym_wrapper_contact_obs_and_penalty():
    """contact_obs appends [links/3, enclosed]; step_penalty shifts
    reward by exactly the penalty on identical trajectories."""
    from env.gym_wrapper import GraspGymWrapper
    env = GraspGymWrapper(seed=0, max_steps=10, contact_obs=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,)
    # No contact yet at reset: contact features are zeros.
    assert obs[-2] == 0.0 and obs[-1] == 0.0
    obs2, _, _, _, info = env.step(np.zeros(3))
    assert obs2.shape == (18,)
    assert obs2[-2] == info["links_touching"] / 3.0
    assert obs2[-1] == (1.0 if info["enclosed"] else 0.0)

    plain = GraspGymWrapper(seed=0, max_steps=10)
    taxed = GraspGymWrapper(seed=0, max_steps=10, step_penalty=0.1)
    plain.reset(seed=0)
    taxed.reset(seed=0)
    for _ in range(5):
        _, r0, _, _, _ = plain.step(np.zeros(3))
        _, r1, _, _, _ = taxed.step(np.zeros(3))
        assert r1 == r0 - 0.1


def test_gym_wrapper_single_object_obs():
    """n_objects=1: no distractors, 10-dim obs (12 with contacts)."""
    from env.gym_wrapper import GraspGymWrapper
    env = GraspGymWrapper(seed=0, n_objects=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)
    env2 = GraspGymWrapper(seed=0, n_objects=1, contact_obs=True)
    obs2, _ = env2.reset(seed=0)
    assert obs2.shape == (12,)


def test_single_object_isolation_is_inf_not_crash():
    """n_objects=1: target_isolation is inf (vacuously isolated),
    and the wrapper steps without crashing."""
    import math
    env = WholeArmGraspEnv(n_objects=1, seed=0)
    env.reset(seed=0)
    env.target_index = 0
    assert math.isinf(env.target_isolation())
    env.step([0.0, 0.0, 0.0])  # must not raise
    from env.gym_wrapper import GraspGymWrapper
    w = GraspGymWrapper(seed=0, n_objects=1)
    w.reset(seed=0)
    _, _, _, _, info = w.step(np.zeros(3))
    assert math.isinf(info["isolation"])


def test_frame_info_aligns_with_frames():
    """run_episode frame_info is 1:1 with frames (incl. singulate
    reverts on crowded seed 7) and its last entry matches the metrics."""
    from perception.grounding import HeuristicLabeller
    from scripts.demo_vlm_grasp import run_episode
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    plan = HeuristicLabeller().plan(env.render(), env.objects,
                                    "grasp the red box in isolation")
    metrics, frames, _ = run_episode(
        env, plan, "grasp the red box in isolation", verbose=False)
    fi = metrics["frame_info"]
    assert len(fi) == len(frames), f"{len(fi)} status vs {len(frames)} frames"
    for s in fi:
        assert set(s) == {"links", "enclosed", "caged", "fc"}
        assert 0 <= s["links"] <= 3
    last = fi[-1]
    assert last["links"] == metrics["links_touching"]
    assert last["enclosed"] == metrics["enclosed"]
    assert last["caged"] == metrics["caged"]
    assert last["fc"] == metrics["force_closure"]


def test_overlay_status_and_gifs_with_info(tmp_path):
    """Status bar draws on frames without changing size; GIF builders
    accept per-frame info (single + side-by-side)."""
    from scripts.make_video import (make_gif, make_side_by_side_gif,
                                    overlay_status, status_text)
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :] = (200, 200, 200)
    status = {"links": 2, "enclosed": True, "caged": True, "fc": False}
    assert status_text(status) == "links:2 enc:Y cage:Y FC:N"
    over = overlay_status(Image.fromarray(frame), status)
    assert over.size == (60, 40)
    assert np.asarray(over).tobytes() != frame.tobytes()
    frames = [frame] * 5
    infos = [{"links": i % 3, "enclosed": False, "caged": False,
              "fc": False} for i in range(5)]
    out = make_gif(frames, {"instruction": "x", "target_color": "red"},
                   tmp_path / "single.gif", fps=15, scale=1.0,
                   frame_info=infos)
    assert out.exists()
    sbs = make_side_by_side_gif(
        frames, frames, "vlm", "heuristic", "cap1", "cap2",
        tmp_path / "sbs.gif", fps=15, scale=1.0,
        left_info=infos, right_info=infos)
    assert sbs.exists()
    # Same geometry as the no-info composite: 60+4+60 wide, 28+40+28 high.
    assert Image.open(sbs).size == (124, 96)


def test_write_html_report(tmp_path):
    """HTML report: aggregates, per-seed plans (escaped), GIF links."""
    from scripts.compare_planners import write_html_report
    rows = [
        {"seed": 0, "planner": "vlm", "planner_error": None,
         "force_closure": False, "caged": False, "preconditions_met": True,
         "total_reward": 1.5, "isolation_before": 0.1,
         "isolation_after": 0.1, "target_index": 0, "target_color": "red",
         "push_away": [], "rationale": "already <isolated> & clear",
         "api_call": True, "tokens": 1153},
        {"seed": 0, "planner": "heuristic", "planner_error": None,
         "force_closure": False, "caged": False, "preconditions_met": True,
         "total_reward": 1.5, "isolation_before": 0.1,
         "isolation_after": 0.1, "target_index": 0, "target_color": "red",
         "push_away": [], "rationale": "scripted", "api_call": False,
         "tokens": 0},
    ]
    gif_dir = tmp_path / "compare-0"
    gif_dir.mkdir()
    (gif_dir / "side_by_side.gif").write_bytes(b"GIF")
    gif_map = {0: {"vlm": str(gif_dir / "vlm.gif"),
                    "heuristic": str(gif_dir / "heuristic.gif"),
                    "side_by_side": str(gif_dir / "side_by_side.gif")}}
    out = write_html_report(rows, gif_map, tmp_path / "report.html",
                            "grasp the red box in isolation",
                            ["vlm", "heuristic"])
    html = out.read_text()
    assert "vlm" in html and "heuristic" in html
    assert "seed 0" in html
    assert "already &lt;isolated&gt; &amp; clear" in html
    assert "compare-0/side_by_side.gif" in html
    assert "0.00" in html  # FC rate 0/1


def test_diffusion_denoiser_and_sampling_shapes():
    """MLP denoiser: noise prediction matches chunk shape; DDIM
    sampling returns a chunk in [-1, 1]."""
    import torch
    from diffusers import DDIMScheduler
    from scripts.diffusion_policy import (DiffusionMLP, sample_chunk)
    torch.manual_seed(0)
    model = DiffusionMLP(obs_dim=16, chunk=16, hidden=64, depth=2)
    noisy = torch.randn(4, 16, 3)
    obs = torch.randn(4, 16)
    t = torch.randint(0, 100, (4,))
    pred = model(noisy, obs, t)
    assert pred.shape == (4, 16, 3)
    sched = DDIMScheduler(num_train_timesteps=100)
    chunk = sample_chunk(model, sched, torch.randn(16), 16,
                         num_steps=5)
    assert chunk.shape == (16, 3)
    assert chunk.min() >= -1.0 and chunk.max() <= 1.0


def test_chunk_dataset_sampling():
    """ChunkDataset samples (obs, chunk) pairs within true lengths."""
    import numpy as np
    from scripts.train_diffusion import ChunkDataset
    obs = np.random.randn(3, 20, 16).astype(np.float32)
    act = np.random.randn(3, 20, 3).astype(np.float32)
    ds = ChunkDataset(obs, act, np.array([20, 10, 5]), chunk=4)
    o, a = ds.sample(8)
    assert o.shape == (8, 16) and a.shape == (8, 4, 3)


def test_collect_scripted_demos_smoke(tmp_path):
    """Scripted collection writes padded demos + metadata (keep=all,
    envelop, 2 seeds — pipeline check, not a quality claim)."""
    from scripts.collect_demos import collect_one
    import json
    obs, act, info = collect_one(0, "envelop", 1, False)
    assert obs.shape[1] == 10  # n_objects=1 -> 10-dim obs
    assert act.shape == (len(obs), 3)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)
    assert set(info) >= {"force_closure", "caged"}


def test_collect_ppo_demo_fc_seed9(tmp_path):
    """PPO demo collection runs on the new spawn, outcome not locked.

    History: old test locked seed 9 FC=False under planar physics
    (old spawn) — the old 2-step pass was a 3D tipping jam, not a
    planar grasp. Under the new spawn (r 0.28-0.40, theta +-2.4)
    the same 500k model gets FC=True in 1 step on seed 9, verified
    as a planar pinch (z fixed, pure yaw, 2 links, 8 wrenches
    spanning, target moved 0.13 m — not a tipping jam). That is a
    coincidental policy/scene interaction, not a regression
    invariant, so this test no longer locks FC True or False.
    It locks the pipeline: returncode 0, a seed-9 row, and
    kept == FC when --keep fc (kept 1/1 if FC, 0/1 if not).
    Do not "fix" a future FC change by restoring freejoint boxes.
    """
    import subprocess
    out = tmp_path / "ppo_demo"
    proc = subprocess.run(
        [sys.executable, "scripts/collect_ppo_demos.py",
         "--model", "models/grasp_ppo_500k", "--seeds", "9",
         "--keep", "fc", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env={**os.environ, "MUJOCO_GL": "osmesa",
             "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "seed 9: FC=" in proc.stdout
    if "seed 9: FC=True" in proc.stdout:
        assert "kept 1/1" in proc.stdout
    else:
        assert "kept 0/1" in proc.stdout


def test_chunk_dataset_fc_anchored_pairs():
    """FC-anchored windows end at the FC step and repeat."""
    import numpy as np
    from scripts.train_diffusion import ChunkDataset
    obs = np.random.randn(2, 10, 16).astype(np.float32)
    act = np.random.randn(2, 10, 3).astype(np.float32)
    ds = ChunkDataset(obs, act, np.array([10, 10]), chunk=4,
                      fc_steps=np.array([5, -1]), anchor_repeat=4)
    anchored = [p for p in ds.pairs if p == (0, 2)]
    # (0, 2) is both a sliding window and the FC-anchored window
    # ([2..5] ends at FC step 5), so it appears 1 + anchor_repeat times.
    assert len(anchored) == 5
    assert (1, 0) in ds.pairs  # sliding windows still present


def test_diffusion_steps_per_epoch_floor():
    """Small demo sets get a floor on steps/epoch, not 2."""
    from scripts.train_diffusion import steps_per_epoch
    assert steps_per_epoch(135, 64, 50) == 50  # the S5 case: was 2
    assert steps_per_epoch(1000, 64, 50) == 50  # floor still binds
    assert steps_per_epoch(5000, 64, 50) == 78  # floor stops binding
    assert steps_per_epoch(6, 64, 50) == 50


def test_diffusion_train_small_data_uses_floor(tmp_path):
    """End-to-end wiring: a 6-pair dataset trains 50 steps/epoch."""
    import subprocess
    demos = tmp_path / "tiny"
    obs = np.random.randn(2, 6, 16).astype(np.float32)
    act = np.random.randn(2, 6, 3).astype(np.float32)
    np.savez(demos.with_suffix(".npz"), obs=obs, actions=act,
             lengths=np.array([6, 6]))
    proc = subprocess.run(
        [sys.executable, "scripts/train_diffusion.py",
         "--demos", str(demos), "--epochs", "1", "--chunk", "4",
         "--out", str(tmp_path / "tiny_model")],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env={**os.environ, "MUJOCO_GL": "osmesa",
             "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "6 pairs, 50 steps/epoch" in proc.stdout
    assert (tmp_path / "tiny_model.pt").exists()


def test_wrapper_max_delta_q_caps_target():
    """A tight cap limits the joint target the wrapper sends."""
    from env.gym_wrapper import GraspGymWrapper
    w = GraspGymWrapper(seed=0, max_delta_q=0.5)
    w.reset(seed=0)
    prev_q = w.env._obs()["qpos"].astype(np.float64)
    seen = {}

    def fake_step(target):
        seen["target"] = np.asarray(target, dtype=np.float64)
        return None, 0.0, False, {"force_closure": False,
                                  "links_touching": 0, "enclosed": False}

    w.env.step = fake_step
    w.step(np.array([1.0, 1.0, 1.0]))
    assert np.all(seen["target"] <= prev_q + 0.5 + 1e-9)
    assert np.all(seen["target"] >= prev_q - 0.5 - 1e-9)
    # Without a cap the full scaled target passes through.
    w2 = GraspGymWrapper(seed=0)
    w2.reset(seed=0)
    w2.env.step = fake_step
    w2.step(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(seen["target"], 2.6)


def test_approach_face_offsets():
    """approach_face: near/far/left/right yaw offsets from the target."""
    import math
    from planning.primitives import approach_face
    # Target on +x axis: base yaw 0.
    assert approach_face((0.5, 0.0), "near") == 0.0
    assert abs(abs(approach_face((0.5, 0.0), "far")) - math.pi) < 1e-9
    assert approach_face((0.5, 0.0), "left") == math.pi / 2
    assert approach_face((0.5, 0.0), "right") == -math.pi / 2
    # Target on +y axis: base yaw pi/2; far wraps into [-pi, pi].
    assert approach_face((0.0, 0.5), "near") == math.pi / 2
    assert approach_face((0.0, 0.5), "far") == -math.pi / 2
    with __import__("pytest").raises(ValueError):
        approach_face((0.5, 0.0), "diagonal")


def test_singulate_effort_sets_push_distance(monkeypatch):
    """A shove commands 0.25 m, a nudge 0.06 m, on the planned object."""
    import scripts.demo_vlm_grasp as demo
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    env.target_index = 0
    calls = []

    def fake_push(env_, start_xy, theta, dist, frames, frame_info=None):
        calls.append((np.asarray(start_xy, dtype=float), theta, dist))

    monkeypatch.setattr(demo, "push", fake_push)
    target_xy = env._obs()["objects"][0][:2]
    obj2_xy = env._obs()["objects"][2][:2]
    demo.scripted_singulate(env, 0, [], push_targets=[{"index": 2,
                                                      "effort": "shove"}])
    assert len(calls) == 1
    start_xy, _, dist = calls[0]
    assert dist == 0.25
    # Push starts at the planned object's target-facing edge, not
    # somewhere unrelated: within ~0.15 m of the object's center.
    assert float(np.linalg.norm(start_xy - obj2_xy)) < 0.15
    assert float(np.linalg.norm(start_xy - target_xy)) > 0.05

    env2 = WholeArmGraspEnv(seed=7)
    env2.reset(seed=7)
    env2.target_index = 0
    calls.clear()
    demo.scripted_singulate(env2, 0, [],
                            push_targets=[{"index": 2, "effort": "nudge"}])
    assert len(calls) == 1 and calls[0][2] == 0.06


def test_different_plans_diverge_on_same_seed():
    """Regression test for the identical-VLM/heuristic videos: two
    plans on the same seed must produce different trajectories, and
    an explicit push_away must not fall back to other objects."""
    from scripts.demo_vlm_grasp import run_episode
    base = {"grasp_target": 0, "rationale": "t"}
    plan_near = {**base, "push_away": [], "approach_side": "near"}
    plan_far = {**base, "push_away": [], "approach_side": "far"}
    env_a = WholeArmGraspEnv(seed=5)
    env_a.reset(seed=5)
    m_a, frames_a, _ = run_episode(env_a, plan_near,
                                   "grasp the red box in isolation",
                                   verbose=False)
    env_b = WholeArmGraspEnv(seed=5)
    env_b.reset(seed=5)
    m_b, frames_b, _ = run_episode(env_b, plan_far,
                                   "grasp the red box in isolation",
                                   verbose=False)
    assert m_a["approach_side"] == "near"
    assert m_b["approach_side"] == "far"
    assert len(frames_a) != len(frames_b) or any(
        not np.array_equal(fa, fb) for fa, fb in zip(frames_a, frames_b))

    # Explicit plan: kept pushes are a subset of the planned objects
    # (no silent fallback to other objects).
    plan_push = {**base,
                 "push_away": [{"index": 2, "effort": "clear"}],
                 "approach_side": "near"}
    env_c = WholeArmGraspEnv(seed=7)
    env_c.reset(seed=7)
    m_c, _, _ = run_episode(env_c, plan_push,
                            "grasp the red box in isolation",
                            verbose=False)
    assert {p["index"] for p in m_c["pushes"]} <= {2}


def test_normalize_plan_legacy_and_invalid():
    """normalize_plan: legacy ints -> clear, missing side -> near,
    unknown effort/side raises."""
    from perception.grounding import normalize_plan
    valid = {0, 1, 2, 3}
    plan = normalize_plan({"grasp_target": 0, "push_away": [1],
                           "rationale": "old"}, valid)
    assert plan["push_away"] == [{"index": 1, "effort": "clear"}]
    assert plan["approach_side"] == "near"
    with __import__("pytest").raises(ValueError):
        normalize_plan({"grasp_target": 0,
                        "push_away": [{"index": 1, "effort": "yeet"}],
                        "approach_side": "near", "rationale": "x"}, valid)
    with __import__("pytest").raises(ValueError):
        normalize_plan({"grasp_target": 0, "push_away": [],
                        "approach_side": "diagonal",
                        "rationale": "x"}, valid)


def test_empty_push_away_means_no_pushes():
    """Regression (2026-10-28 rerun): run_episode with push_away=[]
    must push nothing, even on a crowded seed where the old
    []->None conversion auto-pushed crowding objects (seed 7:
    auto pushed to isolation 0.197; with [] it must stay 0.106)."""
    from scripts.demo_vlm_grasp import run_episode
    env = WholeArmGraspEnv(seed=7)
    env.reset(seed=7)
    plan = {"grasp_target": 0, "push_away": [],
            "approach_side": "near", "rationale": "no preconditions"}
    m, _, _ = run_episode(env, plan, "grasp the red box in isolation",
                          verbose=False)
    assert m["pushes"] == []
    assert m["isolation_before"] == m["isolation_after"]


def test_ledger_key_changes_with_prompt_and_rejects_legacy_plan():
    """Regression (2026-10-28 rerun): cache key covers prompt +
    schema version, and a legacy-shaped stored plan is not served."""
    from perception.ledger import make_key, is_canonical_plan
    objs = [{"index": 0, "color": "red"}, {"index": 1, "color": "blue"}]
    k1 = make_key("m", "grasp the red box", objs, "imghash",
                  prompt="prompt A")
    k2 = make_key("m", "grasp the red box", objs, "imghash",
                  prompt="prompt B")
    assert k1 != k2
    assert make_key("m", "grasp the red box", objs, "imghash",
                    prompt="prompt A") == k1
    # Legacy S5 entry shape: bare ints, no approach_side.
    assert not is_canonical_plan({"grasp_target": 0, "push_away": [2],
                                  "rationale": "old"})
    assert is_canonical_plan({"grasp_target": 0,
                              "push_away": [{"index": 2,
                                             "effort": "clear"}],
                              "approach_side": "near",
                              "rationale": "new"})


def test_boxes_stay_planar_under_hard_push():
    """Regression for 'physics look off': boxes are slide/yaw only,
    so a hard swing cannot tip them, lift them, or change their z.
    Drives the arm through the seed-33 scene with uncapped joint
    targets (the PPO-style jump that previously launched boxes)."""
    import mujoco
    env = WholeArmGraspEnv(seed=33)
    env.reset(seed=33)
    env.target_index = 0
    for q in ([2.6, -2.6, 2.6], [-2.6, 2.6, -2.6], [0, 0, 0]):
        for _ in range(20):
            env.step(np.array(q, dtype=float))
    for o in env.objects:
        pose = env._obs()["objects"][o["index"]]
        # z fixed at the box half-height, quaternion pure yaw
        assert pose[2] == pytest.approx(0.06)
        assert pose[4] == pytest.approx(0.0, abs=1e-9)
        assert pose[5] == pytest.approx(0.0, abs=1e-9)
        # Not launched to tens of metres (old freejoint bug put a
        # box at -20 m). In-plane sliding itself is unbounded by
        # design — planar joints have no table edge — so the new
        # seed-33 scene legitimately slides a box to 2.01 m while
        # staying flat; the invariant is z + pure yaw, not distance.
        assert abs(pose[0]) < 5.0 and abs(pose[1]) < 5.0
    # and the direct MuJoCo body orientation agrees: no x/y quat
    for o in env.objects:
        xquat = env.data.xquat[env._obj_body_id[o["index"]]]
        assert xquat[1] == pytest.approx(0.0, abs=1e-9)
        assert xquat[2] == pytest.approx(0.0, abs=1e-9)
        assert env.data.xpos[env._obj_body_id[o["index"]]][2] == \
            pytest.approx(0.06)

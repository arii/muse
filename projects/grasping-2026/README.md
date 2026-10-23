# Language-Conditioned Grasping in Simulation

Foundation models wired into a real robotics pipeline: an instruction like
"pick up the red mug" flows through language grounding → segmentation →
grasp synthesis → motion planning → execution in MuJoCo. No hardware, no
fine-tuning — the point is the integration.

> This is the **non-thesis** version: a general VLM foundation-model demo and
> portfolio piece (not the 2014 thesis redo). The VLM backend is swappable —
> local open models (Qwen-VL/LLaVA) for the portfolio, or Nemotron via Nebius
> Token Factory if targeting the Nebius hackathon.

## The pipeline

| Stage | Module | What it does | Status |
|---|---|---|---|
| 1. Language grounding | `perception/grounding.py` | VLM: instruction + image → target object | ✅ working (+ offline fallback) |
| 2. Segmentation | `perception/segment.py` | SAM: detection → pixel mask | 🟡 interface fixed, SAM wiring TODO |
| 3. Grasp synthesis | `perception/grasp.py` | mask → ranked antipodal grasps | 🟡 interface fixed, sampler TODO |
| 4. Motion planning | `planning/motion.py` | q_start → q_goal trajectory | ✅ interpolation; RRT TODO |
| 5. Execution | `env/` (MuJoCo) | planar 3-link arm, randomized objects, contact-based reward | ✅ working |

## Quickstart

```bash
pip install -e .
python scripts/demo_vlm_grasp.py --instruction "grasp the red box"
```

Episodes (frames + metadata) land in `episodes/`. Without `NEBIUS_API_KEY`
the grounding stage uses an offline color-matching fallback.

Set `NEBIUS_API_KEY` + `NEBIUS_VLM_MODEL` to run grounding through a
Nemotron vision model on Nebius Token Factory instead of a local VLM.

## Container

```bash
docker build -t grasping-2026 .
docker run --rm -v $PWD/episodes:/workspace/episodes grasping-2026 \
    --instruction "grasp the blue box" --seed 1
```

Headless rendering is via `MUJOCO_GL=osmesa` (set in the image); no GPU or
display needed. The SAM checkpoint (M2) is mounted at runtime with `-v`,
never baked into the image. Secrets go via `-e` (`NEBIUS_API_KEY`,
`NEBIUS_VLM_MODEL`), never into the image.

## Visualizer

Episodes render to animated GIFs (Pillow-only, no extra deps):

```bash
python scripts/demo_vlm_grasp.py --instruction "grasp the red box" --seed 1 --frame-every 1
python scripts/make_video.py episodes/<stamp>/ --out demo.gif --fps 10
```

Each GIF gets a caption bar with the instruction, target color, and whether
the episode caged the object. `--frame-every 1` keeps full-rate frames for
smooth video; the default `4` keeps episode files small. A live interactive
viewer is out of scope — the container story is headless.

### Pinned working versions
Recorded 2026-10-22 from a green dependency resolution
(`pip install ".[rl]"` dry-run; base image `python:3.11-slim`):

| Dep | Version | Notes |
|---|---|---|
| mujoco | 3.16.0 | sim backend; needs `mjGEOM_BOX` enum name (not the removed `mjtGEOM_BOX` alias) |
| numpy | 2.5.4 | |
| pillow | 13.0.0 | |
| requests | 2.34.2 | VLM API client |
| gymnasium | 1.4.0 | `[rl]` extra; satisfies SB3's `gymnasium<2.0` |
| stable-baselines3 | 2.9.0 | `[rl]` extra |
| torch | 2.14.1 | via SB3 (CPU wheel in container) |
| setuptools | 84.2.1 | build backend |

System packages (Debian trixie, both stages): `libglfw3`, `libgl1`,
`libosmesa6`; builder adds `build-essential`.

## What's next

1. Wire SAM into `perception/segment.py` (checkpoint + predictor).
2. Implement antipodal sampling in `perception/grasp.py`.
3. Generalize the sim backend from the planar arm to a 3D tabletop
   arm+gripper (`env/` is the v1 backend; interfaces above don't care).
4. RL baseline (`scripts/train_rl_baseline.py`) for the learned-vs-classical
   comparison.
5. 60-second demo video + public repo.

## Blog post

Draft: `../goals/learn-llm-driven-grasping-tutorial/files/language-conditioned-grasping-blog-draft.md`
(in the grasping goal's files — placement TBD: arii.github.io vs BoomTick).

## Tutorials

- `docs/vlm-grounding-tutorial.md` — the VLM grounding stage worked end to
  end: prompt, request shape, parsing, failure modes, and how to verify it
  (mock tests in CI + `scripts/smoke_vlm.py` against the live model).

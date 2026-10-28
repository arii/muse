# Experiment harness, setup, and local training

## Setup

Local (this VM, no Docker): venv `~/workspace/.m1-venv`, Python 3.12,
mujoco 3.16.0, gymnasium 1.4.0, stable-baselines3 2.9.0,
torch 2.14.1+cpu, diffusers 0.42.1. Headless rendering needs
`MUJOCO_GL=osmesa` and apt `libosmesa6 libglfw3 libgl1` — VM restarts
wipe apt packages, reinstall them if MuJoCo import fails on GL.

S5 (the machine that can run the VLM): checkout
`~/muse/projects/grasping-2026`, `NEBIUS_API_KEY` in `.env`
($50 Builder credits, key never recorded in the repo).
No Docker on this VM; all S5 commands below are Docker-first.

## Run

Single demo (heuristic, no API key needed):

```
docker compose run --build --rm demo --instruction "grasp the red box in isolation" --seed 7
```

Compare VLM vs heuristic, seeds 0-9, with GIFs + HTML + markdown:

```
docker compose run --build --rm --entrypoint python demo \
  scripts/compare_planners.py --seeds 0-9 --gifs --html
```

Outputs: `episodes/compare-<seed>/side_by_side.gif`,
`episodes/compare-report.html`, `episodes/compare-report.md`.
The markdown file is the readable result: aggregates + one line
per seed, errors in full. The HTML's GIFs are linked files and
do not survive print-to-PDF — read the markdown for a PDF-able
version, open the HTML in a browser for the videos.

Pure VLM mode in compare: a VLM failure counts as a failure
(`planner_error`), no heuristic fallback. The demo's own
`--planner vlm` still falls back, deliberately.

## What the metrics mean

- `force_closure`: the goal criterion. Planar wrench test, needs
  >=4 wrenches. A grasp passes only on this.
- `caged`: strict diagnostic. Target inside the arm polygon AND
  loop closed (tip within 0.15 m of the link1 segment). The old
  polygon-only metric gave false positives on open arms.
- `total_reward`: 0.5 x links touching + 2.0 if force closure,
  summed over steps in compare rows. It inflates on long
  non-passing episodes — do not read it as a pass rate.
- `isolation`: min xy distance from target to nearest other box.
  Gate is 0.15 m. A plan that says push nothing and starts
  crowded fails preconditions, it does not auto-push.

## Environment (current, after 2026-10-28 fixes)

- 3-link arm: 0.30 / 0.25 / 0.20 m, j1/j2/j3 clamped +-2.6 rad.
- Boxes: planar slide-x/slide-y/hinge-yaw, z fixed 0.06, no tipping.
- Spawn (enveloping workspace): r 0.28-0.40 m, theta -2.4 to
  2.4 rad. Older runs used r 0.28-0.55, theta -pi to pi and
  spawned impossible tasks — their seed numbers are not
  comparable to current seeds.
- Target is index 0, red. VLM never estimates coordinates:
  it picks target index, push list with effort, and approach
  side; the executor resolves those to geometry.

## Local training (no API key)

PPO and diffusion run on CPU, in the container, no Nebius call:

```
docker compose run --build --rm --entrypoint python demo \
  scripts/train_rl_baseline.py --help
docker compose run --build --rm --entrypoint python demo \
  scripts/train_diffusion.py --help
```

Current status: old PPO/diffusion models are stale (trained on
freejoint physics + old spawn). Retrain before any new
force-closure or diffusion claim. Recommended first retrain:
PPO 100k steps with `--contact-obs --step-penalty 2.0
--max-delta-q 1.0` (penalty must exceed 1.5, the max per-step
contact reward, or touch-and-hold camping returns), then eval
seeds 0-9, then collect FC demos only if eval shows FC > 0.
Diffusion from scripted demos caps at caging — FC demos must
come from PPO rollouts, and the bottleneck is demo count
(15-16 demos memorized, did not generalize; aim for seeds
0-999, ~150 FC demos, before raising epochs).

Local model note: the S5's Ollama models (qwen3:8b/14b) are
text-only and cannot replace the vision VLM. Local work here
means local *training* (PPO/diffusion on CPU) and the offline
heuristic planner, not a local VLM. The VLM path still needs
Nebius Token Factory + a vision model with JSON mode.

## Visualization

- Demo GIF: every frame, full scale, per-frame status overlay
  (links/enc/cage/FC) — bottom strip on single GIFs, top of
  each half on side-by-side GIFs.
- GIF colors: all frames share one palette (see
  `scripts/make_video.py:_save_gif`) so a box color cannot
  shift frame to frame. Red palette is (0.65, 0.02, 0.02),
  rendering (252, 84, 84); the old (0.85, 0.2, 0.2) rendered
  (252, 130, 130), light pink.
- Frame trace: `frame_info` in metrics is 1:1 with GIF frames,
  including reverted pushes (truncated on revert).

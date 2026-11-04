# Hackathon submission — Meta Global AI Developer Hackathon

**Project:** Language-conditioned whole-arm grasping in simulation
**Track:** Indie · **Problem statement:** Push the Frontier
**Demo video:** `video/out/demo_video.mp4` (122 s, < 3 min — upload to
YouTube/unlisted and link on the CV submission page)
**Repo:** public (`github.com/arii/muse`), this directory
(`projects/grasping-2026/`)

## What it is

Tell a three-jointed arm what to grasp, in words. A VLM grounds the
instruction, **SAM 3.1 (Meta Model API) segments the named target from a
text prompt**, an antipodal sampler proposes grasps on the mask, and
MuJoCo executes. Around that spine, the project benchmarks seven
planner families (scripts, PPO, BC, diffusion, VLM) on five clutter
scenes × seeds 0–9.

## Meta models used (requirement)

- **SAM 3.1 via Meta Model API** — the perception stage. Text phrase
  ("red box") → pixel mask, verified live (IoU 0.73 vs simulator
  ground truth) and wired into `scripts/verify_pipeline.py`
  (segmentation + grasp-synthesis stages, figures saved).
- Built with **Muse Code** (this repo was developed with it throughout).

The VLM planner itself runs on Nebius (Nemotron); the Meta-model
component of the system is SAM 3.1 perception.

## Built during the hackathon (Oct 27 – Nov 4, 115 commits)

Pre-existing base (Oct 22–26): repo scaffold, MuJoCo env + scripted
heuristic demo, perception stub files. Everything below is new:

- SAM 3.1 integration: Responses-API client, official-parser SSE
  decoding, recorded-fixture tests, pipeline wiring, grasp figures
- PPO from scratch (40/50): reward shaping studies (enclosure bonus,
  precontact penalty, ablated variants), mixed-scene training
- Cloning: BC + diffusion policy, val-split fair pipeline, chunk
  study, 2× volume studies (BC ties teacher 40–40)
- Task + harness: 5 scene presets, compare harness, full
  scene×planner matrix, VLM ladder/attribution/canary studies
- Artifacts: results site, HF demo-dataset exporter, Remotion videos
- This video (`video/src/demo.tsx`) and its narration

## Run it

```bash
docker build -t grasping-2026 .
docker run --rm -v "$(pwd)":/w -w /w --env-file .env \
  --entrypoint python3 grasping-2026 \
  scripts/verify_pipeline.py --seed 4   # needs META_API_KEY
docker run --rm -v "$(pwd)":/w -w /w --entrypoint python3 \
  grasping-2026:lockedtest -m pytest tests/ -q   # suite, no keys
```

`.env` needs `META_API_KEY` (+ `META_MODEL_ID=sam-3.1`) for live SAM;
tests use a recorded fixture and need no keys.

# Grasping experiment — implementation instructions
Prepared 2026-10-22 for the non-thesis VLM grasping pipeline
(`~/workspace/grasping-2026/`). Docker approach modeled on `arii/BoomTick`.

## 1. Objective

Build a working language-conditioned grasping loop in simulation:
instruction → VLM grounding → SAM segmentation → grasp synthesis →
motion planning → MuJoCo execution, all inside a reproducible Docker
container. The finished experiment (repo + 60-second video) becomes the
material for the blog post draft in this goal's files.

## 2. Current repo state

| Path | State |
|---|---|
| `env/arm.xml`, `env/whole_arm_grasp.py` | ✅ Working: planar 3-link arm, randomized objects, contact reward, top-down render |
| `perception/grounding.py` | ✅ Working: Nebius/Nemotron client + offline heuristic fallback |
| `perception/segment.py` | 🟡 Interface fixed, SAM wiring TODO |
| `perception/grasp.py` | 🟡 Interface fixed, antipodal sampler TODO |
| `planning/motion.py` | ✅ Joint interpolation; RRT TODO |
| `scripts/demo_vlm_grasp.py` | ✅ End-to-end demo with episode logging |
| `scripts/train_rl_baseline.py` | Notes/stub |
| `pyproject.toml` | deps: mujoco, numpy, pillow, requests (+ gym/SB3 optional) |

## 3. Implementation order

**M1 — Containerize (first).** Write the Dockerfile below, `docker build`
it, and run the demo headless inside the container. Nothing else starts
until `python scripts/demo_vlm_grasp.py` produces an episode from inside
the container. This is the reproducibility foundation everything else
builds on.

**M2 — SAM wiring.** `perception/segment.py`: pip-install
`segment-anything`, download a ViT-H checkpoint (document where it lives;
mount it, don't bake weights into the image), implement
`segment(rgb, prompt)` returning a boolean mask. Verify on a demo frame.

**M3 — Grasp sampler.** `perception/grasp.py`: contour extraction →
antipodal pair sampling → width/approach scoring → ranked grasps.
Unit-test on synthetic masks before touching the sim.

**M4 — 3D tabletop env.** Generalize `env/` from the planar arm to a 3D
arm+gripper tabletop scene ("pick up the red mug"). Keep the module
interfaces (`reset/step/render`, reward dict) stable so the pipeline
doesn't care which backend is underneath.

**M5 — Video + repo.** 60-second screen capture of the loop, public repo
(Ari creates it), README with the pipeline table.

## 4. Docker container spec

Model on BoomTick's multi-stage pattern (builder → slim runtime), adapted
for Python + MuJoCo:

```dockerfile
# Stage 1: build/install
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libglfw3 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[rl]"

# Stage 2: runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglfw3 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY env/ ./env/
COPY perception/ ./perception/
COPY planning/ ./planning/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./
ENTRYPOINT ["python", "scripts/demo_vlm_grasp.py"]
```

Notes:
- `MUJOCO_GL=osmesa` = headless CPU rendering. For GPU VLM inference,
  run with `--gpus all` and document it; the container must still work
  on CPU (fallback path).
- SAM checkpoint: mount at runtime (`-v`), never baked into the image.
- Secrets (`NEBIUS_API_KEY`): via `-e` / env file, never in the image.

## 5. CI (GHCR)

Follow the BoomTick pattern — reference its composite action remotely
rather than copying it:

```yaml
- uses: arii/BoomTick/.github/actions/setup-docker-buildx@main
- uses: docker/metadata-action@v5
  with: { images: ghcr.io/arii/grasping-2026 }
- uses: docker/build-push-action@v6
  with:
    push: true
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

## 6. Dependency table

| Dep | Version | Why | Notes |
|---|---|---|---|
| python | 3.11-slim | base image | |
| mujoco | ≥3.2 | sim backend | pip wheel; osmesa for headless |
| numpy, pillow, requests | — | arrays, images, VLM API | |
| segment-anything | git/​pip | SAM stage (M2) | checkpoint mounted, not baked |
| gymnasium, stable-baselines3 | optional `[rl]` | RL baseline | |
| torch (+cuda) | M2+ | SAM / local VLM | `--gpus all` documented |

## 7. Constraints

- The Muse GitHub App **cannot create repos** (403 — lacks the permission).
  Ari creates the GitHub repo himself; the connector can push files once
  it exists. Do not attempt repo creation.
- arii.github.io / boomtick.blog are **read-only** for this work.
- Pin versions when the container first builds green; record the working
  set in the README.

## 8. Definition of done (per milestone)

- M1: `docker build -t grasping-2026 .` succeeds; demo runs headless in
  the container and writes an episode to `episodes/`.
- M2: `segment()` returns a correct mask on a demo frame.
- M3: sampler returns ranked grasps on synthetic masks; top grasp looks
  sane on a real frame.
- M4: pick-and-place demo runs in the 3D env through the full pipeline.
- M5: repo public, video recorded, blog post has its material.

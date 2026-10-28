# Language-conditioned grasping in simulation — reproducible container.
#
# Build:   docker build -t grasping-2026 .
# Run:     docker compose run --rm demo --instruction "grasp the red box in isolation" --seed 7
#          (or raw docker: docker run --rm -v $PWD/episodes:/workspace/episodes grasping-2026)
#          docker run --rm -e NEBIUS_API_KEY -e NEBIUS_VLM_MODEL \
#              -v $PWD/episodes:/workspace/episodes grasping-2026 \
#              --instruction "grasp the blue box"
#
# Training / diffusion / eval run in the SAME container (docker first —
# no local venv needed). The demo service's entrypoint is the demo
# script; override it with --entrypoint python:
#   docker compose run --build --rm --entrypoint python demo scripts/train_grasp.py --timesteps 100000
#   docker compose run --build --rm --entrypoint python demo scripts/eval_policy.py --model models/grasp_ppo --episodes 10 --seeds 0-9
#   docker compose run --build --rm --entrypoint python demo scripts/train_diffusion.py --demos data/demos_ppo_fc --epochs 50 --chunk 8 --out models/diffusion
# Models and demos persist via the ./models and ./data mounts in
# docker-compose.yml (gitignored — copy model zips in separately).
#
# Secrets come from .env via docker-compose.yml (see .env.example).
#
# Notes:
#  - MUJOCO_GL=osmesa gives headless CPU rendering (no GPU/display needed).
#  - Torch is the CPU build (no CUDA, no triton): the default PyPI torch
#    bundles triton, whose import segfaults on CPU-only hosts during PPO
#    setup. Installing CPU torch FIRST means pip keeps it when SB3's
#    torch requirement is resolved.
#  - libgomp1 is torch's OpenMP runtime — required on slim images.
#  - SAM checkpoint (M2): mount at runtime (-v), NEVER baked into the image.
#  - Secrets (NEBIUS_API_KEY): passed via -e / env-file, never in the image.
#
# Stage 1: build/install
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libglfw3 libgl1 libosmesa6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml ./
# CPU torch FIRST (see note above), then the project extra — pip keeps
# the already-installed CPU build when resolving SB3's torch dep.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
# Non-editable install: packages land in site-packages, no fragile
# editable path links between stages.
RUN pip install --no-cache-dir ".[rl]"

# Stage 2: runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglfw3 libgl1 libosmesa6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY --from=builder /usr/local/lib/python3.11/site-packages \
     /usr/local/lib/python3.11/site-packages
COPY env/ ./env/
COPY perception/ ./perception/
COPY planning/ ./planning/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./
ENTRYPOINT ["python", "scripts/demo_vlm_grasp.py"]

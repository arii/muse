# Language-conditioned grasping in simulation — reproducible container.
#
# Build:   docker build -t grasping-2026 .
# Run:     docker compose run --rm demo --instruction "grasp the red box in isolation" --seed 7
#          (or raw docker: docker run --rm -v $PWD/episodes:/workspace/episodes grasping-2026)
#          docker run --rm -e NEBIUS_API_KEY -e NEBIUS_VLM_MODEL \
#              -v $PWD/episodes:/workspace/episodes grasping-2026 \
#              --instruction "grasp the blue box"
#
# Secrets come from .env via docker-compose.yml (see .env.example).
#
# Notes:
#  - MUJOCO_GL=osmesa gives headless CPU rendering (no GPU/display needed).
#  - SAM checkpoint (M2): mount at runtime (-v), NEVER baked into the image.
#  - Secrets (NEBIUS_API_KEY): passed via -e / env-file, never in the image.
#
# Stage 1: build/install
FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libglfw3 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml ./
# Non-editable install: packages land in site-packages, no fragile
# editable path links between stages.
RUN pip install --no-cache-dir ".[rl]"

# Stage 2: runtime
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglfw3 libgl1 libosmesa6 \
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

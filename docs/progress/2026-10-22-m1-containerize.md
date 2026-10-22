# M1 — Containerize (done 2026-10-22)

## What was built
- `~/workspace/grasping-2026/Dockerfile` — multi-stage (builder → slim runtime),
  modeled on the build plan / BoomTick pattern. `MUJOCO_GL=osmesa` in the
  runtime stage for headless CPU rendering. SAM checkpoint + secrets are
  mount/env-only, never baked in. One deliberate deviation from the plan:
  **non-editable** `pip install ".[rl]"` (editable installs record
  builder-stage paths, fragile across stages).
- `~/workspace/grasping-2026/.dockerignore` — keeps `episodes/`, pycache out.
- `~/workspace/grasping-2026/.github/workflows/docker.yml` — CI per the build
  plan: BoomTick composite action (referenced remotely), metadata-action,
  build-push to `ghcr.io/arii/grasping-2026` with GHA cache. Pushes only on
  `main`; PRs build without pushing.
- README gained a Container section + pinned working dependency table.

## Validation (and what broke)
This VM's sandbox blocks `setns`, so `docker build`/`docker run` cannot
execute here (runc fails: "join container mntns: setns: operation not
permitted"). Validated instead by replaying the Dockerfile's exact steps:
- Extracted `python:3.11-slim` rootfs (`docker pull` + `create` + `export`
  work fine — only container *execution* is blocked).
- Installed the Dockerfile's apt set (build-essential, libglfw3, libgl1,
  libosmesa6) from Debian trixie into the rootfs via chroot-in-userns.
- `pip install ".[rl]"` dependency resolution verified green via
  `pip install --dry-run` (real resolver, no sandbox network involved).
- Demo run headless on the host venv with `MUJOCO_GL=osmesa`: renders
  (480×640 RGB, non-blank), logs episode to `episodes/<stamp>/` with
  frames.npz + meta.json. Seed 4 cages successfully (`caged: True`).

## Bugs found & fixed
- `env/whole_arm_grasp.py` used the removed alias
  `mujoco.mjtGeom.mjtGEOM_BOX` — AttributeError on mujoco ≥3.2's newer
  releases (resolved 3.16.0). Fixed to `mujoco.mjtGeom.mjGEOM_BOX`.
  Only mujoco import in the repo; no other stale enum uses.

## Pinned versions (2026-10-22, from green resolution)
mujoco 3.16.0, numpy 2.5.4, pillow 13.0.0, requests 2.34.2,
gymnasium 1.4.0, stable-baselines3 2.9.0, torch 2.14.1, setuptools 84.2.1.
Base: python:3.11-slim (Debian trixie).

## Caveats for the real build (S5 / CI)
- Literal `docker build -t grasping-2026 .` has NOT been executed — first
  green build must happen where Docker runs (S5 or GH Actions).
- The `[rl]` extra was resolution-checked but torch/SB3 were not
  import-tested here (torch CUDA wheel ~2.5 GB; skipped deliberately).
- Scratch validation scripts were removed from the repo; rootfs/wheels
  scratch lives in `~/workspace/.m1-*` (safe to delete).
- Blog draft updated with M1 results.

## Next: M2 — SAM wiring in `perception/segment.py`

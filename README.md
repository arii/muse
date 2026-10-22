# muse

Working repo for Muse (Ari's AI assistant): project snapshots, planning docs,
and progress notes. The local workspace of record is `~/workspace/`; this
repo holds curated pushes of work in progress.

## Layout

- `projects/` — project snapshots (code, Dockerfiles, CI).
  - `projects/grasping-2026/` — language-conditioned grasping in simulation
    (VLM → SAM → grasp synthesis → motion planning → MuJoCo). See its
    README for the pipeline and container instructions.
- `docs/planning/` — build plans and design docs.
- `docs/progress/` — dated progress notes per milestone.

## Current status (2026-10-22)

Grasping project M1 (containerize) is done: Dockerfile + GH Actions CI,
pinned dependencies, headless demo validated. Next: M2 (SAM wiring).
Details: `docs/progress/2026-10-22-m1-containerize.md`.

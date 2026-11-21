# muse

> Unmaintained, experimental, and not graduated. This repository is a historical sandbox, not an active project backlog. Snapshots are preserved for reference. For current work, see [Ariel Anders' portfolio](https://arii.github.io/).

Working repo for Muse (Ari's AI assistant): project snapshots, planning docs,
and progress notes. The local workspace of record is `~/workspace/`; this
repo holds curated pushes of work in progress.

## Layout

- `projects/` — project snapshots (code, Dockerfiles, CI).
  - [Tokens with Taste](https://github.com/arii/boomtick/tree/main/tokens-with-taste) — maintained in arii/boomtick.
  - `projects/grasping-2026/` — language-conditioned grasping in simulation
    (VLM → SAM → grasp synthesis → motion planning → MuJoCo). See its
    README for the pipeline and container instructions.
- `docs/planning/` — build plans and design docs.
- `docs/progress/` — dated progress notes per milestone.

## Historical status (2026-10-22)

Grasping project M1 (containerize) is done: Dockerfile + GH Actions CI,
pinned dependencies, headless demo validated. The recorded next step was M2 (SAM wiring); it is not an active commitment.
Details: `docs/progress/2026-10-22-m1-containerize.md`.

## Graduation criteria

A project leaves this sandbox only when it has a clear scope, an active maintainer, and no dependency on one-off sandbox hacks. Promotion does not happen automatically.

Existing automation must be inventoried before it is disabled or moved; this status note does not change runners, schedules, containers or services.

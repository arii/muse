# Interactive experiment page — plan (deferred)

Goal: host the grasping experiment interactively at `arii.github.io/muse`.
Status: plan only — not started. Tier 1 is cheap and recommended first;
Tier 2 is a separate project; do Tier 2 only if Tier 1 earns it.

## Tier 1 — Interactive episode explorer (static, ~1–2 days)

**What it is:** a static page where a visitor scrubs through a logged
episode: drag a slider, watch the arm move, read what the VLM said and why,
see reward/contacts per frame. "Interactive" = explorable, not live.

**How it works:**
1. **Export step** (`scripts/export_episode.py`, to be written): converts
   `episodes/<stamp>/{frames.npz, meta.json}` into a web bundle:
   `episode.json` (manifest: instruction, VLM rationale, target color,
   per-frame reward / links_touching / caged / qpos) + frames as JPGs
   (smaller than PNG; ~24 frames × 480×640 is tiny).
   - Requires extending episode logging first: the demo currently logs
     per-episode metadata only. Log per-frame `info` dicts (reward,
     links_touching, caged) alongside frames.
2. **Page** (vanilla HTML/JS, no framework): `<img>` or canvas, range-slider
   scrubber, play/pause, caption panel (instruction, target, VLM rationale),
   per-frame readouts (reward, contacts, caged flag).
3. **Hosting:** standalone static files under `/muse/` on arii.github.io —
   no coupling to the site's Vite/React build. (Site source is
   `arii/portfolio`; the static dir bypasses it deliberately.)

**Why Tier 1 first:** zero backend, works on GitHub Pages today, and it's
the artifact that makes a portfolio visitor believe the pipeline is real.

## Tier 2 — Live in-browser sim (weeks, separate project)

**What it is:** the experiment actually running in the visitor's browser.

**How it works:**
- Physics: `mujoco_wasm` (DeepMind's official WebAssembly build of MuJoCo,
  github.com/google-deepmind/mujoco_wasm) — load `env/arm.xml`, step in JS.
- Controller: port the scripted caging phases (later: the grasp pipeline)
  to JS.
- Grounding (the hard part — needs a VLM in the browser):
  - Option A: in-browser SmolVLM via transformers.js (WebGPU, ~2B params,
    runs locally, no API key — best for a public demo).
  - Option B: call the Nebius/Nemotron API from the page (needs a key
    strategy — user-supplied key field, or a tiny serverless proxy; never
    ship a secret in client JS).
- SAM/grasp stages: start simplified (2D mask + scripted controller),
  upgrade as M2/M3 land in Python.

**Risks:** mujoco_wasm API drift vs. Python MuJoCo; model download size
(~GB on first visit); mobile/WebGPU performance; essentially maintaining
two implementations of the pipeline.

## Sequencing

1. Tier 1 whenever there's a free day — it also produces the M5 demo
   assets (the GIF visualizer in `scripts/make_video.py` is the rough cut).
2. Tier 2 only after Tier 1 exists *and* the page is pulling weight
   (traffic, client conversations). Revisit quarterly.

## Open questions (Ari)

- Confirm the arii.github.io read-only exception: is `/muse` wanted?
- Page source: standalone static dir (recommended) vs. a React route in
  `arii/portfolio`?
- Tier 2 grounding: in-browser model (no key, big download) vs. API key
  field (small, needs key management)?

---
name: demo-video
description: Producing a demo video for a repo with the projects/demo-video toolkit
---

# Demo-video toolkit usage

The pipeline lives in `projects/demo-video/` (Remotion 4.0.540, React 19,
Rspack, 1280×720@30, Kokoro-82M offline TTS). One markdown script per video.

## Workflow for a new video

1. Read the target repo's README and key screens; draft a narration script
   from `projects/demo-video/templates/demo-video.md`. Keep chapters short
   (4–7 for a 60–120 s video). Write narration in spoken prose — numbers are
   spelled as read automatically.
2. `npx demo-video plan scripts/<repo>.md` — compiles and validates the
   script into `src/generated/<id>.json` (commit both files; `src/Root.tsx`
   discovers them, no Root edits needed).
3. `npx demo-video render scripts/<repo>.md` — runs plan → tts → render;
   output `out/<id>.mp4` (gitignored). The composition self-sizes: each
   chapter is frame-aligned to its measured narration audio via
   `calculateMetadata` + `getAudioDuration` (the in-repo voiceover.md
   pattern), and captions are sentence-level cues cut from the same script.
   Verify with ffprobe (video + audio stream, 1280×720, 30 fps; per-chapter
   duration within ±0.5 s of the narration file).
4. Human-QA the render: voice listenability (never robotic — this is a named
   acceptance check), caption sync, visual quality.

## Rules

- Never touch `projects/grasping-2026/**` — path filters trigger Python CI
  and Docker builds.
- Never rely on Remotion's runtime browser downloader; use `npm run setup`
  and the pre-fetched chrome-headless-shell (`--browser-executable` is
  resolved automatically by the CLI).
- No API keys or cloud TTS anywhere; kokoro-js is fully offline after the
  model cache is populated.
- `--silent` renders timed silent chapters; a chapter with no narration is a
  timed silent card by design, not a bug.

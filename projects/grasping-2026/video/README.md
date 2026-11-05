# Demo video (Remotion)

Narrated demo video for the Meta hackathon submission.
Composition `demo` in `src/demo.tsx` (122 s, 1280×720, 30 fps).

## Chapters

| # | Frames | Time | Content |
|---|--------|------|---------|
| 1 | 0–300 | 0:00–0:10 | Title card |
| 2 | 300–750 | 0:10–0:25 | Pipeline: 6 stages, snake layout |
| 3 | 750–2100 | 0:25–1:10 | SAM hero: real API mask reveal + split view |
| 4 | 2100–2550 | 1:10–1:25 | Planner lineup (7 GIFs) |
| 5 | 2550–2910 | 1:25–1:37 | PPO-win clip |
| 6 | 2910–3360 | 1:37–1:52 | Results matrix |
| 7 | 3360–3660 | 1:52–2:02 | Close |

Chapters 4–6 reuse the existing clip components (`lineup.tsx`,
`ppo-win.tsx`, `matrix.tsx`).

## Build

Prereqs: Node 24. Install deps once:

```bash
npm i --loglevel=error
```

Render:

```bash
npx remotion render demo demo_video.mp4 --browser-executable=<path-to-chrome-headless-shell>
```

Chromium note: Remotion's built-in Chromium downloader can time out on
slow networks. If it does, download Chrome headless shell manually
(Remotion prints the exact expected version + URL on failure) and pass
it via `--browser-executable`. The bundled headless shell is the tested
path; don't substitute a system Chrome.

Preview a single frame without rendering:

```bash
npx remotion still demo frame.png --frame=600 --browser-executable=<path>
```

Or run the interactive preview: `npm run dev`.

## Assets (`public/`)

- `*.gif` — planner episode clips (checked in).
- `sam_frame.png` — full-frame MuJoCo top-camera render (seed 2,
  640×480) fed to SAM 3.1: the whole arm and all boxes visible (the
  old 320×240 center crop showed the arm base as a close-up with
  links running off-frame).
- `sam_mask.png` — binary mask returned by the live SAM 3.1 call
  (full frame).
- `sam_overlay.png` — mask composited over the frame (vivid green
  fill). Frame/mask are `verification/20261104_155532/10_sam_frame.png`
  and `11_sam_mask.png` (IoU 0.77 vs sim ground truth by the
  `verify_pipeline._red_gt` rule); regenerate with
  `scripts/verify_pipeline.py` (needs META_API_KEY) and compositing
  as above.

## Captions (no voiceover)

Burned-in captions replace the old TTS narration: each chapter
`<Sequence>` in `src/demo.tsx` mounts `<Captions>` (`src/captions.tsx`)
with an inline cue array (chapter-relative ms, `Caption`-shaped).
Cue text is verbatim from `VO_SCRIPT.md` (now the caption script);
long lines are paged across cues, never rewritten. The render has no
audio track by design — verify with:

```bash
ffprobe -v error -show_entries stream=codec_type out/demo_video.mp4
```

which should list video only.

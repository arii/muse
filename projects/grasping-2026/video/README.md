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
- `sam_frame.png` — MuJoCo top-camera render (seed 2) fed to SAM 3.1,
  center-cropped 320×240 at 2× for legibility (same pixels).
- `sam_mask.png` — binary mask returned by the live SAM 3.1 call (same crop).
- `sam_overlay.png` — mask composited over the frame (vivid green + box).
  Mask/overlay came from the verified 2026-11-04 API response
  (IoU 0.73 vs sim ground truth, full-frame); regenerate with the
  trio script (needs META_API_KEY) and compositing as above.
- `vo/*.mp3` — narration, one track per chapter, wired via `<Audio>`
  inside each `<Sequence>` (audio starts at the sequence start).
  Generated with `/opt/hatch/bin/tts`:

```bash
/opt/hatch/bin/tts speak --output public/vo/title.mp3 --text-stdin <<'EOF'
Tell a three-jointed arm what to grasp, in words. It figures out the rest.
EOF
```

  (Repeat per chapter; the VO lines live in the demo script at
  `workspace/goals/learn-llm-driven-grasping-tutorial/files/meta-demo-script.md`.)
  Numbers are spelled out as spoken ("seventy-seven percent",
  "forty of fifty"). Replace with your own voiceover by dropping in
  same-named MP3s.

## If the render has no audio

Verify the output before blaming the player:

```bash
ffmpeg -i demo_video.mp4 -af volumedetect -f null - 2>&1 | grep max_volume
```

Speech peaks around −8 to −12 dB. If a chapter is silent, check that
its MP3 exists in `public/vo/` and that the `<Audio>` tag sits inside
that chapter's `<Sequence>`.

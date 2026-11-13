# QA note: hrm proof video

- Script: `scripts/hrm.md` · Render: `out/hrm.mp4` (gitignored; delivered as a project file)
- Toolkit head at render: see PR (branch `video/hrm`)

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 92.5 s (`ffprobe` format.duration = 92.501) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, r_frame_rate 30/1) |
| Audio stream | Present: aac |
| Audio/video drift | Sum of narration wavs 92.53 s vs video 92.501 s — within 0.03 s |
| Per-chapter narration files | 7 wav files under `public/voiceover/hrm/` (tts log: 10.0–15.7 s each, 28.6 s synthesis per 100 words) |
| Numbers spelled as read | Plan output verified: "145 beats per minute" → "one hundred forty-five beats per minute"; "76 percent" → "seventy-six percent" |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/hrm.md` is the single source) |
| No `projects/grasping-2026/**` touched | Commit contains only `projects/demo-video/**` files |

## Content provenance (the repo's own material)

- `screenshot` chapters use two captures taken this session from the running app (dev server on 127.0.0.1:3000, mock client streaming): the `/client/mock` mock streamer form (mock-client.png) and the live dashboard with the mock athlete at 145 BPM / 76% / FAT BURN zone (dashboard-live.png), both 2880×1800.
- `terminal` chapter shows the dev server's real startup log lines (`Spotify Polling Service Initialized.`, `ConnectionMonitor started.`, `> Ready on http://127.0.0.1:3000`) captured this session.
- Narration facts from the repo README: Next.js + custom Express server, persistent WebSocket `/ws`, Tabata timer service, Spotify polling, zone tiles, Playwright VRT, PM2 deploys, Strava write-only compliance.

## Visual quality (verified by frame inspection)

Five frames sampled (8 s, 33 s, 47 s, 62 s, 88 s) and inspected this session:

- `title` cards (open/close): gold kicker "HRM DASHBOARD" over large white headline; clean hierarchy.
- `screenshot` chapters: image framed in a themed panel with slow zoom; gold chapter heading; no clipping at 1280×720.
- `bullets` chapters ("What it does", "Built to run for real"): legible, no overflow.
- `terminal` chapter: real log lines, `$ pnpm run dev` prompt in green.
- Motion consistent with the toolkit's bezier ease-out language.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown matches the sentence being delivered (at 33 s the mock-client cue reads "set an age and a target of one hundred forty-five beats per minute"; at 47 s the dashboard-live cue matches the tile description being spoken).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice, 28.6 s synthesis per 100 words, model cache verified by preflight (all four checks PASS). No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The pinned `af_heart` voice's sample (`fl_BehVsFQD`, PR #6 evidence) is still awaiting human listenability review — the same pending sign-off recorded on PRs #8–#11. If a human listener judges the voice robotic, this video's voice verdict reverses and re-synthesis (toolkit `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

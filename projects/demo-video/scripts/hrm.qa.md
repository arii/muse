# QA note: hrm proof video

- Script: `scripts/hrm.md` · Render: `out/hrm.mp4` (gitignored; delivered as a project file)
- Toolkit head at render: see PR (branch `video/hrm-intro` — intro-credit rework on top of PR #12)

## Intro credit rework (this branch)

The "Meet HRM" opening chapter adds the line "Frontier SF — made by Ariel Anders using obvious.ai." — spoken by the narration, shown as the title-card subtitle, and captioned verbatim when delivered (one script source, per toolkit invariant). All other chapters are verbatim from the merged PR #12 cut. Raw `obvious.ai` TTS pronunciation is not human-audited (same caveat as the boomtick/glossy intro reworks); the tech-dancer rework instead spells it as "obvious dot AI" — if a human listener judges the raw token awkward, re-authoring that one sentence and re-running tts → render is the fix path.

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 97.1 s (`ffprobe` format.duration = 97.067) — inside 60–120 s |
| Intro credit | Spoken, title-card subtitle, and caption bar all carry "Frontier SF — made by Ariel Anders using obvious.ai." (frame inspection at 14.5 s; cue 12.8–16.4 s in `src/generated/hrm.json`) |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, r_frame_rate 30/1) |
| Audio stream | Present: aac |
| Audio/video drift | Sum of narration wavs 97.10 s vs video 97.067 s — within 0.04 s |
| Per-chapter narration files | 7 wav files under `public/voiceover/hrm/` (intro `meet-hrm.wav` is 18.0 s including the credit sentence; re-synthesis this session shifted non-intro chapter pacing by ≤2.6 s each — narration text there is verbatim from PR #12; 35.4 s synthesis per 100 words) |
| Numbers spelled as read | Plan output verified: "145 beats per minute" → "one hundred forty-five beats per minute"; "76 percent" → "seventy-six percent" |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/hrm.md` is the single source) |
| No `projects/grasping-2026/**` touched | Commit contains only `projects/demo-video/**` files |

## Content provenance (the repo's own material)

- `screenshot` chapters use two captures taken this session from the running app (dev server on 127.0.0.1:3000, mock client streaming): the `/client/mock` mock streamer form (mock-client.png) and the live dashboard with the mock athlete at 145 BPM / 76% / FAT BURN zone (dashboard-live.png), both 2880×1800.
- `terminal` chapter shows the dev server's real startup log lines (`Spotify Polling Service Initialized.`, `ConnectionMonitor started.`, `> Ready on http://127.0.0.1:3000`) captured this session.
- Narration facts from the repo README: Next.js + custom Express server, persistent WebSocket `/ws`, Tabata timer service, Spotify polling, zone tiles, Playwright VRT, PM2 deploys, Strava write-only compliance.

## Visual quality (verified by frame inspection)

Five frames sampled (8 s, 33 s, 47 s, 62 s, 88 s) and inspected this session:

- `title` cards (open/close): gold kicker "HRM DASHBOARD" over large white headline; clean hierarchy; the open card's subtitle ends with the Frontier SF credit line (verified by frame inspection at 14.5 s this session).
- `screenshot` chapters: image framed in a themed panel with slow zoom; gold chapter heading; no clipping at 1280×720.
- `bullets` chapters ("What it does", "Built to run for real"): legible, no overflow.
- `terminal` chapter: real log lines, `$ pnpm run dev` prompt in green.
- Motion consistent with the toolkit's bezier ease-out language.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown matches the sentence being delivered (at 14.5 s the caption bar carries "Frontier SF — made by Ariel Anders using obvious.ai." while the credit is spoken; at 47 s the dashboard-live cue matches the tile description being spoken).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice, 28.6 s synthesis per 100 words, model cache verified by preflight (all four checks PASS). No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The pinned `af_heart` voice's sample (`fl_BehVsFQD`, PR #6 evidence) is still awaiting human listenability review — the same pending sign-off recorded on PRs #8–#11. If a human listener judges the voice robotic, this video's voice verdict reverses and re-synthesis (toolkit `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

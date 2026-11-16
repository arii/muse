# QA note: boomtick proof video

- Script: `scripts/boomtick.md` · Render: `out/boomtick.mp4` (gitignored; delivered as a project file)
- Toolkit head at render: see PR (branch `video/boomtick-intro`, intro credit rework)
- Intro credit (this rework): opening chapter adds the line "Frontier SF — made by Ariel Anders using obvious.ai." — spoken in narration and shown verbatim on the opening title card; all other chapters unchanged.

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 93.4 s (`ffprobe` format.duration = 93.440) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, avg_frame_rate 30/1) |
| Audio stream | Present: aac |
| Per-chapter narration files | 6 wav files under `public/voiceover/boomtick/`, each ffprobe-verified valid audio |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/boomtick.md` is the single source) |
| No `projects/grasping-2026/**` touched | `git status` clean except toolkit files |

## Visual quality (verified by frame inspection)

Six frames sampled at chapter midpoints (5.7 s, 18.7 s, 35.9 s, 56.1 s, 73.5 s, 84.7 s) and inspected this session:

- `title` cards (open/close): clean hierarchy — gold kicker, large white headline, dim subtitle.
- `bullets` chapters ("Three pillars", "Rescue pull requests, safely"): bullets legible, no overflow at 1280×720.
- `terminal` chapter shows the repo's real documented commands (`./setup-agent.sh`, `td mcp install --execute`, `node mcp/dist/index.js`); `split` chapter shows the real `pnpm run impact:*` review commands.
- Motion consistent with the toolkit's bezier ease-out language; no clipped text observed in any sampled frame.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown at each sampled timestamp matches the sentence being spoken at that point (e.g. at 35.9 s the caption reads "One command registers it with Claude Code, Cursor, and VS Code." while that sentence is mid-delivery; at 73.5 s the split chapter's first sentence is displayed).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice, 23.3 s synthesis per 100 words. No API keys, no network at synthesis time.

The automated evidence above covers duration, stream validity and cue sync.
Subjective naturalness and pronunciation were not assessed for this render.

## Intro-credit rework note (2026-11-13)

The credit line is authored verbatim as "Frontier SF — made by Ariel Anders using obvious.ai." so the title card and captions show the exact requested string. TTS reads the raw "obvious.ai" via Kokoro's own text normalization; its pronunciation was not independently assessed. Frame inspection at 4 s and 10 s confirms the credit renders on the opening title card.

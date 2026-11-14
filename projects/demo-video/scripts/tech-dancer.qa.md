# QA note: tech-dancer proof video

- Script: `scripts/tech-dancer.md` · Render: `out/tech-dancer.mp4` (gitignored; delivered as a project file)
- Toolkit head at render: branch `video/tech-dancer-intro` (intro-credit rework, PR against `release/demo-video-toolkit`)

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 84.8 s (`ffprobe` format.duration = 84.779) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, r_frame_rate 30/1) |
| Audio stream | Present: aac, full 84.779 s |
| Per-chapter narration files | 8 wav files under `public/voiceover/tech-dancer/`, each ffprobe-verified valid pcm_f32le audio (14.28 / 8.05 / 5.85 / 9.00 / 14.43 / 13.03 / 10.63 / 9.45 s) |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/tech-dancer.md` is the single source) |
| No `projects/grasping-2026/**` touched | `git status` clean except toolkit files |

## Visual quality (verified by frame inspection)

Four frames sampled across the timeline (5.0 s, 23.3 s, 43.3 s, 66.7 s) and inspected this session:

- `title` card (open): clean hierarchy — gold kicker "BOOMTICK", large white headline, dim subtitle; caption bar carries the verbatim opening sentence. **Intro credit (this rework):** the opening narration now ends with "Frontier SF — made by Ariel Anders, using obvious dot AI." — shown on the title card as the subtitle and spoken by the voice; the caption bar carries it verbatim when that sentence is delivered (verified at ~12 s frame). "obvious dot AI" follows the toolkit's spelled-as-read convention (cf. "boomtick dot blog") so the caption stays verbatim with the TTS input; "obvious.ai" as a raw token is not verified TTS-safe.
- `split` chapter ("Built like a product"): narration text beside the `pnpm build` pipeline panel; line lengths safe at 1280×720, no clipping.
- `terminal` chapters ("Meet the td CLI", "Guarded by quality gates"): the repo's own commands (`td doctor`, `td preflight`, `pnpm ci:local`) with the check names `td doctor` actually verifies (node runtime, pnpm, git safe directory, GitHub app credentials) and the repo's `ci:local` gate list (oxlint + eslint, vitest + playwright, semgrep + knip, lighthouse). Note: `td doctor` fails in this sandbox only from a Node version mismatch (expects 24.16.0, sandbox runs 20.20.2); the video shows the tool's real check surface, not this environment's error output.
- Motion consistent with the toolkit's bezier ease-out language; captions legible in every sampled frame.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown at each sampled timestamp matches the sentence being spoken at that point (e.g. at 23.3 s the caption reads the full "Every build runs a typed pipeline…" sentence while that sentence is mid-delivery in the split chapter).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice, 25.2 s synthesis per 100 words. No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The pinned `af_heart` voice's sample (`fl_BehVsFQD`, PR #6 evidence) is still awaiting human listenability review. If a human listener judges the voice robotic, this video's voice verdict reverses and a re-synthesis (voice change within the toolkit's `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

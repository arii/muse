# QA note: posture-guard proof video

- Script: `scripts/posture-guard.md` · Render: `out/posture-guard.mp4` (gitignored; delivered as a project file `fl_IM4ekv7D`)
- Toolkit head at render: see PR (branch `video/posture-guard`)

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 94.4 s (`ffprobe` format.duration = 94.400) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, 30/1) |
| Audio stream | Present: aac, 94.400 s |
| Per-chapter narration files | 8 wav files under `public/voiceover/posture-guard/`, TTS-reported durations 4.85–16.10 s each |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/posture-guard.md` is the single source) |
| No `projects/grasping-2026/**` touched | `git status` clean except toolkit files |
| Toolkit checks | `npm run lint` clean, `npm test` fail 0 |

## Visual quality (verified by frame inspection)

Frames sampled at 8 s, 45 s, and 90 s and inspected this session:

- `title` cards (open/close): clean hierarchy — gold kicker "POSTURE GUARD", large white headline, dim subtitle, caption bar clear of the card text.
- `bullets` chapter ("Never nudges on bad data"): three repo-grounded bullets (depth band, visibility floor, NO PERSON badge).
- `terminal` chapter shows the repo's own documented preflight-report shape and commands (`./setup.sh`, `make run`); `split` chapters show the repo's test-nudge curl and real `config.yaml` knobs (`head_forward_deg: 12.0`, `voice_repeat_cooldown_s: 30.0`, both from `config.yaml` at commit `41d1759`).
- One defect found and fixed this session: the "nudge ladder" split chapter's narration text overflowed behind the caption bar at 45 s; the narration was trimmed and the video re-rendered. Post-fix frame shows all text clear of the caption bar.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown at each sampled timestamp matches the sentence being spoken at that point (at 45 s, mid-chapter, the caption reads "Ten seconds later the spoken prompt fires, repeating at most every thirty seconds." while that sentence is mid-delivery). Number handling is the toolkit's spelled-as-read normalization over the script's own digits.

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, default pinned `af_heart` voice, 26.0 s synthesis per 100 words. No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The same `af_heart` listenability question is already pending on the boomtick and tech-dancer videos (PRs #8, #9). If a human listener judges the voice robotic, this video's voice verdict reverses and a re-synthesis (voice change within the toolkit's `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

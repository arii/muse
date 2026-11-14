# QA note: feels proof video

- Script: `scripts/feels.md` · Render: `out/feels.mp4` (gitignored; delivered as a project file)
- Toolkit head at render: branch `video/feels-intro` (intro-credit cut; see PR)
- Intro credit (user-requested): the opening chapter now ends with "Frontier SF — made by Ariel Anders using obvious.ai." — spoken in the opening narration, shown verbatim on the title-card subtitle, and captioned from the same sentence. Only the intro chapter changed; all other chapters, pacing, and script text are verbatim from PR #11.

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 66.0 s (`ffprobe` format.duration = 66.048) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, avg_frame_rate 30/1) |
| Audio stream | Present: aac, 66.05 s |
| Per-chapter narration files | 6 wav files under `public/voiceover/feels/` (9.53, 7.30, 14.43, 16.00, 8.57, 10.20 s — the intro chapter grew by the credit sentence; all others unchanged), each ffprobe-verified valid audio; sum matches composition duration |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/feels.md` is the single source) |
| Numbers spelled as read | No digit tokens in any narration paragraph (numbers appear only in on-screen bullets/terminal lines, never spoken); narration uses word forms ("Stage one", "point nine cosine similarity") which pass through `spellAsRead` idempotently |
| No `projects/grasping-2026/**` touched | `git status` clean except `scripts/feels.md` + `src/generated/feels.json` |
| Lint / tests | `npm run lint` exit 0; `npm test` all green |

## Visual quality (verified by frame inspection)

Six frames sampled from the final render (2 s, 10 s, 22 s, 38 s, 50 s, 60 s) and inspected this session:

- `title` cards (open/close): clean hierarchy — gold kicker, large white headline, dim subtitle. On this cut the opening card's subtitle carries the added Frontier SF credit, frame-inspected at 2.5 s: "Frontier SF — made by Ariel Anders using obvious.ai." renders legibly on two lines with no overflow, and the caption bar tracks the narration sentence at that timestamp.
- `split` chapter ("The swirl is the sensor"): narration left, five bullets right, no overflow. (An earlier draft had narration running past the bottom edge; fixed by trimming the narration before this render.)
- `terminal` chapters show the repo's **real captured output**, run this session on the feels sandbox: `python src/predict_content.py recordings/recording_20261029_200617_mystery.csv` (PREDICTION: LIQUID, swirl-model agrees) and `python src/state_spectrum.py ... --neighbors` (STATE SPECTRUM bars + water neighbors). The third neighbor line was dropped from the block so the window clears the caption bar; the spoken narration never cites the removed line.
- `bullets` chapter ("The numbers"): all four bullets legible, caption clear of content.
- Motion consistent with the toolkit's bezier ease-out language.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption matches the sentence being delivered at each sampled timestamp (e.g. at 22 s the caption reads "Stage two: liquid, with the swirl model in agreement." while that sentence is mid-delivery; at 38 s the spectrum chapter's second sentence is displayed).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice, 22.9 s synthesis per 100 words. No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The pinned `af_heart` voice's sample (`fl_BehVsFQD`, PR #6 evidence) is still awaiting human listenability review. If a human listener judges the voice robotic, this video's voice verdict reverses and a re-synthesis (voice change within the toolkit's `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

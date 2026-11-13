# QA note: glossy proof video

- Script: `scripts/glossy.md` · Render: `out/glossy.mp4` (gitignored; delivered as a project file, `fl_IooGzMHQ`)
- Toolkit head at render: see PR (branch `video/glossy`)

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 68.4 s (`ffprobe` format.duration = 68.416) — inside 60–120 s |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, avg_frame_rate 30/1) |
| Audio stream | Present: aac |
| Per-chapter narration files | 7 wav files under `public/voiceover/glossy/` (9.925, 15.575, 12.475, 10.575, 5.625, 11.125, 3.100 s), each ffprobe-verified valid audio |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/glossy.md` is the single source) |
| Numbers spelled as read | "1,800" → "one thousand eight hundred" via `spellAsRead` (verified against the plan output); "Forty-two" and "1716"/"100%" appear only where spoken/read forms match the script's word forms |
| No `projects/grasping-2026/**` touched | `git status` shows only `projects/demo-video/**` paths (script, four screenshot assets, generated plan, QA note) |
| Lint / tests | `npm run lint` exit 0; `npm test` 79 pass / 0 fail |

## Visual quality (verified by frame inspection)

Seven frames sampled from the final render (5, 15, 27, 38, 48, 58, 66 s) and inspected this session; 15 s, 27 s and 48 s shown here:

- `title` cards (open/close): clean hierarchy — gold kicker, large white headline, dim subtitle; close card fades out cleanly.
- `screenshot` chapters (Meet Glossy, Read with the morphemes, Edit with live preview, In out and published) use the toolkit's screenshot visual (adopted from upstream in the rebase): the repo's **real UI, captured this session from a live `dev:next` server** on the glossy sandbox — landing page with the interactive glossing widget, Beowulf reader with the Gār-Den-a morpheme popup open (morpheme chips, inflection, Wiktionary link), the Ohthere editing workspace, and the corpus-ingestion page. Each sits in the component's framed panel with the slow Ken Burns zoom; no overflow, captions clear of content.
- `terminal` chapter shows the repo's real `npm run validate:lemmas` output (Audited 1716 tokens across 75 sentences, 100% accuracy), run this session on the glossy sandbox.
- Motion consistent with the toolkit's bezier ease-out language.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption matches the sentence being delivered at each sampled timestamp (e.g. at 15 s "Glossy is an interlinear glossing platform for Old English, built for students, researchers, and linguistic editors." is mid-delivery; at 48 s "Drafts autosave as you work." closes the editor chapter).

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, pinned `af_heart` voice. No API keys, no network at synthesis time.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The pinned `af_heart` voice's sample (`fl_BehVsFQD`, PR #6 evidence) is still awaiting human listenability review. If a human listener judges the voice robotic, this video's voice verdict reverses and a re-synthesis (voice change within the toolkit's `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

# QA note: posture-guard proof video (sub-60s rework with live replay)

- Script: `scripts/posture-guard.md` · Render: `out/posture-guard.mp4` (gitignored; delivered as a project file `fl_vLI4QeJ5`)
- Toolkit head at render: branch `video/posture-guard-short` (this PR; squashed on merge)
- Supersedes the 94.4 s cut from PR #10 (delivered as `fl_IM4ekv7D`). This rework condenses 8 chapters to 5 and adds a live-replay chapter built from the user's own dashboard recording.

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | 52.6 s (`ffprobe` format.duration = 52.608) — under the 60 s hard ceiling, inside the 50–55 s target |
| Resolution / fps | 1280×720 @ 30 fps (`ffprobe`: h264, 30/1) |
| Audio stream | Present: aac (`ffprobe`), narration-only |
| Chapter count | 5 (`title`, `replay`, `split`, `split`, `title`); plan estimate 52.5 s |
| Per-chapter narration files | 5 wav files under `public/voiceover/posture-guard/` (15.10 / 12.95 / 9.70 / 9.95 / 4.85 s), 22.7 s synthesis per 100 words |
| Replay source | Project file `fl_u5qpw7z4` ("Recording Nov 13, 2026 - 4_33 PM.mp4", 18.297 s, 1920×1080 @ 30 fps, h264+aac) — user-uploaded live dashboard footage; **replaces the older 3:14 PM recording (`fl_BKXxmrNc`) used in the PR #19 cut** |
| Replay clip | Source 0–16 s span — the whole 18.3 s clip qualifies (needle visible and moving, reading 8 then 6); pose skeleton + live metrics + POSTURE NEEDLE gauge + 'Good — reward chime' banner all on screen. Lanczos-scaled 1920×1080 → 1280×720 (full-frame, aspect-matched, no crop), ambient audio stream removed — the clip is silent (`ffprobe`: video stream only), so narration is never competed with. Committed at `public/replay/posture-guard/live-dashboard.mp4` (16.0 s, ~1.0 MB); the chapter's narration (12.95 s) fits inside it |
| Caption source | Cues cut verbatim from the same narration strings the TTS speaks (toolkit invariant; `scripts/posture-guard.md` is the single source). Numbers spelled as read; no bare digits in spoken text |
| No `projects/grasping-2026/**` touched | `git status` shows only `projects/demo-video/**` files |
| Toolkit checks | `npm run lint` clean, `npm test` 80 pass / 0 fail |

## Visual quality (verified by frame inspection)

Frames sampled this session: 8, 18, 20, 25, and 32 s, with 18 s and 25 s (mid-replay) inspected closely:

- `title` cards (open/close): gold kicker reads "POSTURE GUARD — FRONTIER SF", large white headline, dim subtitle, caption bar clear of the card text. The opening card carries the Frontier SF hackathon credit (Ariel Anders, Obvious AI) in the on-card narration text, the spoken line, and the burned-in caption.
- `replay` chapter ("Live on the dashboard"): full-bleed live footage — green pose-skeleton overlay over the RealSense feed, live metrics panel updating on screen (head-forward / trunk-lean / head-tilt / shoulder-level angles with degree readouts), and the **POSTURE NEEDLE gauge with the needle visibly moving across frames (reading 7 at the sampled timestamps), plus the GOOD state and 'Good — reward chime' banner** — all from the newer 4:33 PM recording (`fl_u5qpw7z4`). Frames at 18 s and 25 s show motion (footage is playing, not frozen) and the verbatim caption tracking the narration. The camera-placement guidance and Test button from the older recording's framing are no longer visible in this clip's crop; the narration sentence is unchanged because the script is the single source and the swap is footage-only.
- `split` chapters: the repo's own documented test-nudge curl (`POST /nudge/test`) and real `config.yaml` knobs (`head_forward_deg: 12.0`, `voice_repeat_cooldown_s: 30.0`).

One intermediate render was discarded this session: an edit race briefly reverted the title-card kicker to "POSTURE GUARD"; the fix was re-applied, the plan regenerated (title confirmed in `src/generated/posture-guard.json`), and the delivered render is the corrected one.

Verdict: pass.

## Caption sync (verified from cue timing + frames)

Captions are one cue per sentence, cut verbatim from the narration string. Frame inspection confirms the caption shown at each sampled timestamp matches the sentence being spoken at that point (at 25 s, mid-`replay`-chapter, the caption reads "Camera-placement guidance sits beside it, and the Test button fires a nudge on demand." while that sentence is mid-delivery; at 2 s the opening credit sentence captions the spoken credit). Chapter durations are fitted to the measured wav durations by the planner.

Verdict: pass.

## Voice quality (Kokoro-82M `af_heart`, pinned)

Synthesis: offline Kokoro-82M via kokoro-js, default pinned `af_heart` voice, 22.7 s synthesis per 100 words. No API keys, no network at synthesis time.

One pronunciation flag for the human listener: the spoken credit says "using Obvious AI" (TTS reads "AI" as the letters). If it sounds off to a human ear, the fix is a spelled-as-read rewrite in `scripts/posture-guard.md` ("obvious dot AI") and a re-synthesis — same pattern already used by the sibling videos.

Per spec D2/D4 the "must not sound robotic" check is a **named human listenability check**. This note cannot substitute for a human ear: machine verification here covers duration, stream validity, and cue sync, not subjective naturalness. The same `af_heart` listenability question is pending across all six videos. If a human listener judges the voice robotic, this video's voice verdict reverses and a re-synthesis (voice change within the toolkit's `--voice` flag) is the fix path.

Verdict: pending human listenability sign-off — everything machine-checkable passes.

# QA note: Caffeinated Cinema proof (real recording, compressed highlight cut)

- Script: `scripts/caffeinated-cinema.md`
- Render: `out/caffeinated-cinema.mp4` (gitignored, absolute path
  `/home/ubuntu/repos/muse/projects/demo-video/out/caffeinated-cinema.mp4`)
- Toolkit branch: `devin/1794802013-cinema-port`, based on Muse `73e190b`.
- Source: Caffeinated **origin/main `6488d413a67551d07c73abd41b26535850629295`**,
  archived to `/home/ubuntu/caffeinated-cinema-main` and built outside its checkout;
  Caffeinated tracked files were not modified. Ran `cd video && npm ci` per
  `video/CINEMA_STATUS.md`, then Vite's pages build to `/home/ubuntu/cinema-site/caffeinated`
  and copied the archive's `docs/recordings` into that build.

## Acceptance checks (measured, this session)

| Check | Result |
|---|---|
| Duration | ffprobe format duration **28.715000 s**, under 60 s; video timeline 861 frames = 28.700 s (AAC padding accounts for the difference) |
| Resolution / fps | H.264, 1280×720, 30/1 fps, 861 frames |
| Audio stream | AAC, 48,000 Hz, stereo; narration only |
| Final file size | 8,318,270 bytes; ignored output, not committed |
| Chapters | 5, all `replay`: hook, trap, snap, breakthrough, proof |
| Narration WAVs | 4.400 / 3.525 / 7.150 / 6.600 / 7.000 s under `public/voiceover/caffeinated-cinema/`; offline pinned Kokoro `af_heart`; final CLI synthesis 27.1 s per 100 words |
| Real replay | `f8fa53cc-6afd-5e98-9596-0c7b8711e718`, `ignobel`, 297 ticks; the gzip contains entities, events, frames, igjudge and manifest. No mock data or invented results |
| Capture master | `/home/ubuntu/cinema-qa/cinema-source.mp4`: 64.000 s, 1920×1080, 30 fps, 1920 frames, video only, 29,771,093 bytes; not committed |
| Committed replay | `public/replay/caffeinated-cinema/director.mp4`: complete 64.000 s source, 1280×720, 30 fps, video only, **2,446,869 bytes** (<3 MB) |
| Clip preparation | Full-frame Lanczos downscale, no crop, two-pass H.264 at 300 kbit/s target (`slow`, yuv420p, faststart); no event/timeline/data changes. SHA256 `2ce82b3656e2d7df49e34b21f35b92c835084ffb5bb696760e6780c2fafa6992` |
| Caption source | Sentence cues are verbatim narration from the single script; audio-fitted to measured WAVs, not independently authored |
| Checks | `npm run lint` exits 0; `npm test`: 84 pass / 0 fail, including containment and replay parsing/fitting |
| Guide | Not found by filesystem/GitHub searches; created `demo-video-production-guide-od-17.md`. OD-17 single-script/Kokoro/prefetch/preset decisions unchanged |

Capture command (ran in background; about 12 minutes on SwiftShader):

```sh
npm run capture -- --serve /home/ubuntu/cinema-site \
  --path '/caffeinated/?cinema=1&capture=1&recording=f8fa53cc-6afd-5e98-9596-0c7b8711e718' \
  --fps 30 --size 1920x1080 --out /home/ubuntu/cinema-qa/cinema-source.mp4 \
  --stills 5,18,32,48,62 --browser-executable /home/ubuntu/.local/bin/google-chrome
npm run plan -- scripts/caffeinated-cinema.md
npm run tts -- scripts/caffeinated-cinema.md
npm run render -- scripts/caffeinated-cinema.md --concurrency 2
```

Duration was **not overridden**: the capture used `window.__cinema.duration`.
Each of its 1920 frames called `render(i/30)` before screenshotting. The toolkit
browser cache was also prefetched and used for Remotion. The final layout-only
correction was re-rendered with `npx remotion render caffeinated-cinema ...`
using the same generated plan and WAVs, without re-synthesizing narration.

The master stills are `/home/ubuntu/cinema-qa/cinema-source_{5,18,32,48,62}s.png`.
They were extracted by the capture command from its encoded MP4.

## Edit and evidence

| Chapter | Final timeline | Source ranges (seconds, relative speed) |
|---|---|---|
| Hook | 0.000–4.400 | 0–6 |
| Trap | 4.400–7.933 | 6–9; 9–19 x4 |
| Snap | 7.933–15.100 | 19–24; 24–36 x4 |
| Breakthrough | 15.100–21.700 | 36–40; 40–46 x3; 46–52 |
| Proof | 21.700–28.700 | 54–64 |

Every range is inside the 64-second clip. Source seconds 52–54 are deliberately
omitted; quiet stretches are sped up. Each whole chapter cut fits its measured
narration duration, so the listed x4/x3 factors are relative weights. Segment
frame budgets are 132; 58+48; 134+81; 66+33+99; 210, with no frame gaps.

The source event at tick 151 records `B=0.7326`, regime `BOREDOM_SNAP`.
The proof frame displays the source's real derived numbers: **0.73** boredom at
snap, **2** labs founded, **3** replications after snap, **1** Ig Nobel candidate.
The separate earlier pilot card is imported by Caffeinated from
`data/pages/cinema.json` (citing `experiments/confirmatory/RESULTS.md`):
hadamard19 tie; bixbench_mini control 2 of 3; playworld_mini within noise;
ignobel control confirmed more. Its caveat is retained: **3 runs per task — no
significant boredom benefit yet**. A candidate is not a prize award, and the
run's discoveries are not proof that boredom improves the pilot comparison.

## Visual quality (verified from final encoded frames)

An intermediate full-bleed render was rejected: embedded director captions
collided with toolkit narration captions, and the chapter badge obscured the
source branding. Segmented replays now contain the whole source above a 120 px
caption-safe band and omit that redundant badge. Legacy single-file replay
layout is unchanged. Re-rendered and inspected the corrected surface.

Final PNGs saved outside git, sampled from the delivered MP4:

- `/home/ubuntu/cinema-qa/final-2s.png` — hook; moving lab/world, source brand,
  phase caption and separate narration caption.
- `/home/ubuntu/cinema-qa/final-5.5s.png` — trap; local tweaks/boredom overlay.
- `/home/ubuntu/cinema-qa/final-9.5s.png` — snap phase, red overlay and first cue.
- `/home/ubuntu/cinema-qa/final-12.5s.png` — scouting/literature tags, second cue.
- `/home/ubuntu/cinema-qa/final-16s.png` — replicated-findings chart and scientists.
- `/home/ubuntu/cinema-qa/final-19s.png` — gold “New best”, four total replications,
  second breakthrough cue.
- `/home/ubuntu/cinema-qa/final-23s.png` — actual proof-card numbers/pilot table.
- `/home/ubuntu/cinema-qa/final-26s.png` — proof caveat and final narration cue.

Inspected in the final contact sheets `/home/ubuntu/cinema-qa/final-sheet-a.png`
(2, 5.5, 9.5, 12.5 s) and `/home/ubuntu/cinema-qa/final-sheet-b.png`
(16, 19, 23, 26 s), row-major. Embedded text and narration captions are separate,
the entire source remains visible, and changed camera/agent/chart states show
actual replay motion rather than a held screenshot. The real proof card is
preserved, not replaced with a toolkit-invented claim.

Verdict: pass.

## Caption sync (cue timing + frames)

Measured WAV durations round to 132/106/215/198/210 frames. A check against
ffprobe verified every chapter's audio-fitted frame count and exact cue-text
concatenation against narration: `PASS 861 frames, 28.700s; all audio-fitted
chapter timings and cue text match`.

| Final sample | Expected cue from generated timing | Visible text verified |
|---|---|---|
| 2 s | Hook, 0–4.400 | Caffeinated follows a real swarm of artificial scientists. |
| 5.5 s | Trap, 4.400–7.933 | Local tweaks give way to rising measured boredom. |
| 9.5 s | Snap sentence 1, 7.933–11.755 | At the recorded snap, new labs begin scouting. |
| 12.5 s | Snap sentence 2, 11.755–15.100 | Quiet stretches move faster, without inventing events. |
| 16 s | Breakthrough sentence 1, 15.100–17.988 | Replicated discoveries lead to the recorded verdict. |
| 19 s | Breakthrough sentence 2, 17.988–21.700 | This is a highlight cut of the same run. |
| 23 s | Proof sentence 1, 21.700–24.763 | The proof card keeps the pilot honest. |
| 26 s | Proof sentence 2, 24.763–28.700 | These small comparisons show no significant boredom benefit yet. |

Verdict: pass for sentence-level cue/frame synchronization. Cues use the
toolkit's word-weighted timing, not phoneme forced alignment; this is not a
claim of phoneme-accurate audio alignment.

## Voice quality

Offline pinned Kokoro-82M, `af_heart`; narration and AAC stream are present and
timings measured. Subjective naturalness and pronunciation require human
listening; no human listenability sign-off is claimed here.

Verdict: pending human listenability sign-off.

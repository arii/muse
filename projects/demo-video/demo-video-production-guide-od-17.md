# OD-17 production guide: deterministic Cinema and compressed highlight cuts

## Locked decisions (unchanged)

One Markdown script remains the source for narration, captions and chapters.
Use pinned offline Kokoro TTS, the existing prefetch/setup flow and existing
Remotion render presets. This port adds capture and replay edit decisions; it
does not replace the pipeline, choose another voice service or change presets.

## Start from evidence, not a storyboard's imagined results

Choose one actual recording. Record its ID, source commit, event timestamps,
provenance and limitations in the QA note. Never fabricate missing results or
present a pilot as a confirmed benefit. Caffeinated's source recording is
`docs/recordings/f8fa53cc-6afd-5e98-9596-0c7b8711e718.json.gz` (297 ticks).
Its proof card imports `data/pages/cinema.json`, which cites the earlier pilot
in `experiments/confirmatory/RESULTS.md`: **no significant boredom benefit yet**.
The recorded run and the pilot comparison are distinct pieces of evidence.

## The compressed highlight-cut pattern

| Phase | Job | Caffeinated's original source seconds |
|---|---|---|
| hook | Show the product and ask the question before explaining | 0–6 |
| trap | Establish local repetition/rising measured boredom | 6–20 |
| snap | Show the recorded change of regime and real scouting | 20–36 |
| breakthrough | Give discoveries/verdict enough time to read | 36–54 |
| proof | End with actual measured results and their limitations | 54–64 |

Source: [`src/engine/cinema.ts`](https://github.com/arii/caffeinated/blob/main/src/engine/cinema.ts)
(`CinemaPhase`, `CINEMA_PHASES`, `cinemaPlan`, `cinemaTickAt`). It derives
beats from real events and assigns less screen time to quiet ticks. Do not
blindly copy the phase times to another product: derive its beats from its own
recording. Use the [README replay syntax](README.md#replay-highlight-cuts) for
explicit source-clip segments.

Ranges are source seconds, end-exclusive. `x4` gives a quiet stretch one quarter
of its natural screen time relative to `x1` stretches. Segments play in listed
order; gaps skip footage, and deliberate repeats/reordering are allowed. Keep
those editorial choices clear in QA. The complete cut is fitted proportionally
to the chapter's measured narration duration, so nominal `x4` is a relative
edit weight, not a promise of exactly four-times playback after fitting.
Frame budgets use cumulative rounding and exactly fill the chapter. Check
that every range falls inside the actual clip with ffprobe; the planner can
validate syntax and ordering within a range, not remote media duration.
Source endpoints are rounded to composition frames before fitting; rates use
that same trimmed frame span. Sub-frame source ranges fail during planning.
Single-line replay blocks retain their original untrimmed, normal-speed behavior.

Hold important action at readable speed. Accelerate idle/search/waiting
stretches, not the snap, decisive result, or proof card. Short captions make
the replay legible: Caffeinated flashes each phase caption for about **2.8 s**
(`CAPTION_SECONDS`), then fades it, except for the final title. This is a visual
pattern, not a reason to fork the toolkit's single-source narration captions.

## Director mode and capture contract

* `?cinema=1`: hide the ordinary panels, drawers, controls and in-world UI;
  keep the canvas plus intentional director overlay.
* `?cinema=1&capture=1`: stop the wall-clock loop and expose
  `window.__cinema = { ready, duration, render(t) }`. Ready means recording,
  assets and scene are loaded. Duration and `t` are seconds. `render(t)` can
  return a promise; all scene state, camera, lighting, overlays and effects must
  be a pure function of `t`. A call in any order must reproduce that instant;
  no accumulating physics, unseeded randomness or `performance.now()` animation.
* Capture calls `render(i/fps)` and awaits its completion before each screenshot.
  The ffmpeg pipe uses stream backpressure. Slow SwiftShader changes wall time,
  not video pacing. Capture duration defaults to the page's duration.

Capture defaults are 30 fps, 1280×720, and a 30,000 ms readiness timeout
(`--timeout-ms`). Dimensions must be even. `--stills 5,18` makes lossless PNG
screenshots beside the MP4 after separate `render(5)` and `render(18)` calls;
it does not decode stills from the lossy video. Browser selection is explicit
`--browser-executable`, then `CHROME_PATH`, then the toolkit cache. Run setup
to prefetch the cache; `playwright-core` does not download another browser.
Segmented replays contain the whole source above a 120 px narration-caption
band and omit the redundant toolkit badge; legacy clips remain full-frame.

Provenance in Caffeinated:

* [`video/CINEMA_STATUS.md`](https://github.com/arii/caffeinated/blob/main/video/CINEMA_STATUS.md):
  real recording, beat/tick map, resumption commands and known slow-render limits.
* [`video/scripts/cinema.mjs`](https://github.com/arii/caffeinated/blob/main/video/scripts/cinema.mjs):
  frame-stepped Chromium/JPEG→ffmpeg capture, backpressure and sampled stills.
* [`src/engine/cinema.ts`](https://github.com/arii/caffeinated/blob/main/src/engine/cinema.ts):
  pure time/event plan, five phases, weighting, 64 s duration and 2.8 s captions.
* [`src/engine/cinemaView.ts`](https://github.com/arii/caffeinated/blob/main/src/engine/cinemaView.ts):
  director DOM, eased camera, lighting, overlays, real proof-card data.
* [`src/cinema.css`](https://github.com/arii/caffeinated/blob/main/src/cinema.css):
  canvas/overlay-only presentation; original layout targets 1920×1080.
* [`src/main.ts`](https://github.com/arii/caffeinated/blob/main/src/main.ts), Cinema
  flag handling/startCinema/live loop (approximately lines 57–632): capture
  bridge, recording load and exclusion of wall-clock animation in capture mode.
* [`tests/cinema.test.js`](https://github.com/arii/caffeinated/blob/main/tests/cinema.test.js):
  deterministic phase/tick, boredom and replicated-discovery invariants.

## Production and QA checklist

1. Build the actual app in its static/replay mode. Serve build output, not a
   screenshot or invented replay. Capture at its intended layout dimensions.
2. `npm run capture -- --serve <build-root> --path '/?cinema=1&capture=1' --out <clip.mp4>`.
   Server is loopback-only and rejects resolved-path escapes, including symlinks.
   Use `--url` for an already-running app. Missing contract fails explicitly.
3. Put a small real clip under `public/replay/<id>/`; keep large capture masters
   outside git. Record any scaling/re-encoding and byte size. Never modify the
   underlying run to fit a claim.
4. Write the one script; run plan, TTS and render. Fit the edit to measured audio
   through the existing chapter timing, not independently timed caption text.
5. ffprobe the final file: under 60 s for a short proof, 1280×720, 30 fps, audio
   present. Inspect sampled **encoded frames**, including edit boundaries,
   action and proof. Save PNGs outside git and list paths in the QA note.
6. Check cue start/end frames, chapter offsets, narration WAV durations and
   visible caption text together. Cue timing is sentence/word-weighted, not
   phoneme forced alignment: do not call that phoneme-perfect lip sync. Measure
   final audio levels, speech/cue coverage and (when available) ASR word accuracy;
   report those results without claiming subjective naturalness.

Worked example: `scripts/caffeinated-cinema.md` and its `.qa.md` note.

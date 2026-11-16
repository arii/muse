# demo-video

A reusable, offline pipeline that turns a repo plus a narration script into a
polished 1–2 minute demo video: natural voice (Kokoro-82M), burned-in captions,
consistent motion. No API keys anywhere.

Seeded from the proven `projects/grasping-2026/video/` configuration:
Remotion 4.0.540, React 19, TypeScript 5.9, Rspack, 1280×720 @ 30 fps,
`Easing.bezier(0.23, 1, 0.32, 1)` motion language, system-ui font stack.

## Setup

```bash
npm install
npm run setup
```

`npm run setup` prints a preflight report and exits non-zero on failure:

```
PASS node: v20.20.2
PASS ffmpeg: ffmpeg 7.1.5-...
PASS browser: .../chrome-headless-shell
PASS kokoro model: fetched onnx-community/Kokoro-82M-v1.0-ONNX
```

The browser step runs `remotion browser ensure` first; if the documented
downloader timeout hits, it extracts the Chrome for Testing URL/version
Remotion prints and downloads chrome-headless-shell into `.cache/browser/`
manually. Renders always pass `--browser-executable` (or the CLI resolves the
pre-fetched binary itself) — the toolkit never relies on Remotion's runtime
downloader. The kokoro model is cached under `.cache/kokoro`; synthesis is
fully offline afterwards.

## The script is the interface

A narration script is a markdown file. One script drives narration audio,
captions, and scene timing — they cannot drift.

```markdown
---
id: my-repo          # composition id (letters, digits, hyphens)
title: My Repo
---

## A chapter title            # each ## chapter becomes a <Sequence>

<!-- visual: title|bullets|terminal|split -->

Plain paragraphs are narration: spoken by TTS and shown as captions.
Numbers are spelled as read automatically (77% → "seventy-seven percent").

- bullet lines are shown by the bullets/split visuals

```bash
$ fenced blocks become terminal lines
```
```

Visuals: `title` (full-screen card), `bullets` (list entering one by one),
`terminal` (terminal window with typed lines), `split` (text beside a
terminal/panel), `screenshot` (a framed image from the featured repo, read
from `public/screenshots/<script-id>/<chapter-slug>.png` with a slow zoom —
a missing file fails the render explicitly). A chapter with no narration renders as a timed silent card —
explicit, not accidental.

## CLI

```bash
npm run plan -- templates/demo-video.md      # print scene plan JSON + write src/generated/<id>.json
npm run tts -- templates/demo-video.md       # synthesize narration only
npm run render -- templates/demo-video.md    # plan -> tts -> render
```

`render` runs all three stages in one command — it always writes the plan to
`src/generated/<id>.json` itself, so a standalone `plan` run is optional.

(The `demo-video` bin also exists for packing; within the repo, `npm run`
is the interface — npx would resolve the npm registry package of the same
name, not this private one.)

Shared flags: `--script <path>`, `--voice <id>` (default `af_heart`),
`--out <path>`, `--concurrency <n>`, `--browser-executable <p>`,
`--silent` (render timed silent chapters, skip TTS), `--dry-run`, `--help`.

Output defaults to `out/<composition-id>.mp4` (gitignored).

## Adding a video for a new repo

1. Copy `templates/demo-video.md` to `scripts/<repo>.md` and write the
   narration from the repo's README and a content pass.
2. `npm run plan -- scripts/<repo>.md` — this writes
   `src/generated/<id>.json`. Commit both files.
3. `npm run render -- scripts/<repo>.md` — done. `src/Root.tsx` discovers
   generated plans automatically; no Root edits, so parallel video PRs never
   conflict.

## Development

```bash
npm run lint   # eslint (src, scripts, tests) + tsc, zero errors expected
npm test       # node:test unit tests (script parsing, CLI flags, cue timing, preflight logic)
npm run dev    # Remotion Studio
```

## How timing works

Each `##` chapter becomes a `<Sequence>`. Composition sizing follows the
in-repo `remotion-markup/voiceover.md` pattern: `calculateMetadata` measures
each chapter's narration file with `getAudioDuration` (mediabunny) and sizes
the composition — and the chapter — to the audio, frame-aligned at 30 fps.
Captions are one cue per sentence, cut verbatim from the same narration
string the TTS speaks, so audio and text cannot drift. A chapter with no
narration renders as an explicit timed silent card; a narration file that
exists but cannot be measured fails the render with a named error.

`projects/grasping-2026/**` is off-limits to this toolkit: files there trigger
the Python CI and Docker builds.

## Cinema: deterministic capture

```sh
npm run capture -- --url 'http://127.0.0.1:5175/?cinema=1&capture=1' \
  --fps 30 --size 1280x720 --out public/replay/my-demo/director.mp4 --stills 5,18
# Or serve a static build, on an ephemeral 127.0.0.1 port only:
npm run capture -- --serve /path/to/build --path '/?cinema=1&capture=1' \
  --out public/replay/my-demo/director.mp4
# The bin wrapper also accepts: demo-video capture <same flags>
```

`?cinema=1` hides ordinary UI, keeping canvas + director overlay.
`?cinema=1&capture=1` exposes `window.__cinema = { ready, duration, render(t) }`;
all state must be a pure function of `t` in seconds. Capture awaits readiness
and renders each frame explicitly. Browser: `--browser-executable` →
`CHROME_PATH` → toolkit cache. See `npm run capture -- --help` for flags and the
[production guide](demo-video-production-guide-od-17.md) for defaults, lossless
stills, containment, Caffeinated source references and production/QA guidance.

## Replay highlight cuts

Keep the existing single filename in a `replay` fence for normal-speed footage,
or append source-time segments:

````markdown
```replay
clip.mp4
0-6
6-40 x4
40-52
```
````

The filename alone keeps legacy playback. Segment fitting, relative speeds,
validation and caption-safe layout are detailed in the
[highlight-cut guide](demo-video-production-guide-od-17.md#the-compressed-highlight-cut-pattern).

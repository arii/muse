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
terminal/panel). A chapter with no narration renders as a timed silent card —
explicit, not accidental.

## CLI

```bash
npm run plan -- templates/demo-video.md      # print scene plan JSON + write src/generated/<id>.json
npm run tts -- templates/demo-video.md       # synthesize narration only
npm run render -- templates/demo-video.md    # plan -> tts -> render
```

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
npm test       # node:test unit tests (script parsing, CLI flags, preflight logic)
npm run dev    # Remotion Studio
```

Chapter durations and captions come from the plan; once voiceover audio
exists for a chapter (from `npx demo-video tts`), the next PR wiring audio
durations will size those chapters to the audio per the in-repo
`remotion-markup/voiceover.md` pattern.

`projects/grasping-2026/**` is off-limits to this toolkit: files there trigger
the Python CI and Docker builds.

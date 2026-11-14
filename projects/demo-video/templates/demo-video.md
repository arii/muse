---
id: demo
title: Demo Video Toolkit
---

# Demo Video Toolkit

## One command, one video

<!-- visual: title -->

Demo video turns a repository and a narration script into a polished, minute-long demo.

## What is in the box

<!-- visual: bullets -->

- One markdown script drives narration, captions, and timing
- Offline narration with kokoro, no API keys
- Chapter components with a consistent motion language

Everything is derived from a single source, so the captions and the voice never drift.

## Watch the pipeline

<!-- visual: terminal -->

The CLI can run every stage alone.

```bash
$ npx demo-video plan templates/demo-video.md
$ npx demo-video tts templates/demo-video.md
$ npx demo-video render templates/demo-video.md
```

Each stage is checkable on its own before the next one runs.

## Two views, one script

<!-- visual: split -->

The same script can show a terminal beside the narration, or bullets beside it.

```bash
$ npx demo-video render --silent
render: out/demo.mp4
```

This chapter shows a split layout, with the command on the right.

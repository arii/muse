---
id: posture-guard
title: Posture Guard
---

# Posture Guard demo

## A camera that watches your back

<!-- visual: title -->

Posture Guard is a Linux desktop posture corrector. It watches you through an Intel RealSense depth camera, scores how you sit, and nudges you before the slump becomes the afternoon.

## What it actually scores

<!-- visual: bullets -->

- Posture scored from head and shoulder geometry, gated by depth
- A chime first, then a spoken prompt while the slouch persists
- A local web dashboard with the live skeleton overlay

Scoring is pure geometry: head forward, trunk lean, head tilt, and shoulder level angles, checked against a depth gate. A local dashboard shows the live skeleton, so you can verify tracking and tune camera placement.

## From zero to dashboard

<!-- visual: terminal -->

Getting running is two commands, and setup ends with a hardware preflight report.

```bash
$ ./setup.sh
PASS  camera detected — RealSense D415/D435
PASS  USB connection type — 3.x
PASS  depth stream readable
PASS  audio sinks — USB sound card
PASS  TTS renderable
$ make run
dashboard: http://127.0.0.1:8000
```

Setup is idempotent, and every preflight line should read pass before the first run. Then one command serves the dashboard.

## The nudge ladder

<!-- visual: split -->

Hold a slouch for two seconds and the chime fires once. Ten seconds later the spoken prompt fires, repeating at most every thirty seconds. Sit up straight for three seconds and everything goes quiet.

```bash
$ curl -X POST http://127.0.0.1:8000/nudge/test
# chime plays, then the spoken prompt
```

## Never nudges on bad data

<!-- visual: bullets -->

- The person must sit inside the depth band to be scored
- Landmark visibility must clear a floor before a score is trusted
- Leave the desk, and the badge reads NO PERSON — silent

Nudges never fire on bad data. If landmark confidence or depth drops, the badge reads unreliable, and the system stays silent.

## Tuning, not code

<!-- visual: split -->

Every threshold, duration, and audio sink lives in one config file. Chimes too eager? Raise the angle thresholds. Voice too chatty? Stretch the cooldown. All durations are seconds, and all angles are degrees.

```yaml
# config.yaml
scoring:
  head_forward_deg: 12.0
state:
  voice_repeat_cooldown_s: 30.0
```

## No camera? No problem

<!-- visual: split -->

Without a RealSense attached, the pipeline falls back to a synthetic frame source, so the dashboard is fully verifiable before the camera is ever plugged in. The full test suite runs the same way.

```bash
$ pytest
$ ruff check .
```

## Sit like you mean it

<!-- visual: title -->

Posture Guard: a camera that watches your back, so you can sit like you mean it.

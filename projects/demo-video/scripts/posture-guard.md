---
id: posture-guard
title: Posture Guard — Frontier SF
---

# Posture Guard demo

## A camera that watches your back

<!-- visual: title -->

Posture Guard is a Linux desktop posture corrector, built by Ariel Anders for the Frontier SF hackathon using Obvious AI. An Intel RealSense depth camera scores how you sit, and nudges you before the slump sticks.

## Live on the dashboard

<!-- visual: replay -->

Here it is live: the pose skeleton overlaid on the camera feed, and posture metrics updating in real time. Camera-placement guidance sits beside it, and the Test button fires a nudge on demand.

```replay
live-dashboard.mp4
```

## The nudge ladder

<!-- visual: split -->

Hold a slouch and the chime fires first. Ten seconds later, a spoken prompt follows. Sit up straight for three seconds, and everything goes quiet.

```bash
$ curl -X POST http://127.0.0.1:8000/nudge/test
# chime plays, then the spoken prompt
```

## Tuning, not code

<!-- visual: split -->

Every threshold, duration, and audio sink lives in one config file. Raise an angle threshold, stretch the voice cooldown. Tuning, not code.

```yaml
# config.yaml
scoring:
  head_forward_deg: 12.0
state:
  voice_repeat_cooldown_s: 30.0
```

## Sit like you mean it

<!-- visual: title -->

Posture Guard: a camera that watches your back, so you can sit like you mean it.

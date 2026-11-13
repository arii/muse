---
id: hrm
title: HRM Dashboard
---

# HRM Dashboard

## Meet HRM

<!-- visual: title -->

HRM is a real-time heart rate monitoring dashboard for group workouts. It streams live heart rate from Bluetooth sensors to a shared big-screen view, and keeps every phone, laptop, and display in sync.

## What it does

<!-- visual: bullets -->

- Real-time heart rate over WebSockets
- Tabata timer with countdown beeps
- Color-coded HR zone tiles
- Spotify playback controls

The dashboard streams heart rate over WebSockets, runs a Tabata timer with countdown beeps, renders color-coded zone tiles for every athlete, and integrates Spotify so the workout always has a soundtrack.

## Mock client

<!-- visual: screenshot -->

You do not need a sensor to try it. The built-in mock client simulates a Bluetooth heart rate strap: set an age and a target of 145 beats per minute, pick a zone, and start a continuous stream.

## Dashboard live

<!-- visual: screenshot -->

The dashboard picks the stream up instantly. Each athlete gets a large tile showing their live percentage, beats per minute, and calories, color coded by training zone. Here the mock athlete sits at 76 percent, in the fat burn zone.

## Under the hood

<!-- visual: terminal -->

Under the hood, a custom Express server hosts the Next.js app, a persistent WebSocket server, and long-running background services. That stateful design is what keeps every connected screen perfectly in sync.

```bash
$ pnpm run dev
INFO: Spotify Polling Service Initialized.
INFO: ConnectionMonitor started.
INFO: > Ready on http://127.0.0.1:3000
```

## Built to run for real

<!-- visual: bullets -->

- Playwright visual regression suite
- PM2 production deploys
- Strava write-only compliance

The repo ships production muscle: a Playwright visual regression suite guards every screen, PM2 handles deploys, and Strava integration is write-only, so workout data stays local.

## Try HRM

<!-- visual: title -->

HRM, real-time heart rate monitoring for real workouts. Clone the repo, start the mock client, and watch the room light up zone by zone.

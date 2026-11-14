import { Easing, interpolate } from "remotion";

// The motion language of every chapter component: strong ease-out on all
// entrances (linear reads mechanical). Bezier carried forward from
// projects/grasping-2026/video/src/demo.tsx.

export const easeOut = Easing.bezier(0.23, 1, 0.32, 1);

export const fade = (frame: number, start = 0, len = 24) =>
  interpolate(frame, [start, start + len], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });

export const rise = (frame: number, start = 0, len = 24, dist = 28) =>
  interpolate(frame, [start, start + len], [dist, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });

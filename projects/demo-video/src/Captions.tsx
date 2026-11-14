import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { FONT, INK, LINE, PANEL } from "./theme";
import { easeOut } from "./motion";
import type { Cue } from "./script-types";

// Burned-in captions. Cue shape matches @remotion/captions `Caption`
// (text/startMs/endMs); timings are chapter-relative milliseconds and are
// derived from the same script file as the narration, so text and voice
// cannot drift. Each chapter mounts one <Captions> inside its own
// <Sequence>, so trimming or moving a chapter carries its captions with it.
// Carried forward from projects/grasping-2026/video/src/captions.tsx.

export const Captions: React.FC<{ cues: Cue[] }> = ({ cues }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = (frame / fps) * 1000;
  const cue = cues.find((c) => t >= c.startMs && t < c.endMs);
  if (!cue) {
    return null;
  }
  const opacity = Math.min(
    interpolate(t, [cue.startMs, cue.startMs + 150], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
      easing: easeOut,
    }),
    interpolate(t, [cue.endMs - 150, cue.endMs], [1, 0], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    })
  );
  const dy = interpolate(t, [cue.startMs, cue.startMs + 200], [8, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  return (
    <div
      style={{
        position: "absolute",
        bottom: 22,
        width: "100%",
        display: "flex",
        justifyContent: "center",
        opacity,
        transform: `translateY(${dy}px)`,
        pointerEvents: "none",
      }}
    >
      <div
        style={{
          fontFamily: FONT,
          fontSize: 29,
          color: INK,
          backgroundColor: PANEL,
          border: `2px solid ${LINE}`,
          padding: "8px 26px",
          borderRadius: 12,
          maxWidth: 1120,
          textAlign: "center",
          lineHeight: 1.35,
        }}
      >
        {cue.text}
      </div>
    </div>
  );
};

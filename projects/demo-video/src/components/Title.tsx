import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { BG, DIM, FONT, GOLD, INK } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";

// Full-screen chapter card: script title as a kicker, chapter name as the
// headline. Used for `title` chapters and as the fallback for `screenshot`
// chapters until a repo's own assets are wired in.

export const Title: React.FC<{ chapter: Chapter; kicker?: string }> = ({
  chapter,
  kicker,
}) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        backgroundColor: BG,
        alignItems: "center",
        justifyContent: "center",
        fontFamily: FONT,
        padding: 60,
      }}
    >
      {kicker ? (
        <div
          style={{
            fontSize: 26,
            color: GOLD,
            letterSpacing: 4,
            textTransform: "uppercase",
            opacity: fade(frame, 5),
            transform: `translateY(${rise(frame, 5)}px)`,
          }}
        >
          {kicker}
        </div>
      ) : null}
      <div
        style={{
          fontSize: 84,
          fontWeight: 800,
          color: INK,
          textAlign: "center",
          lineHeight: 1.15,
          marginTop: kicker ? 26 : 0,
          opacity: fade(frame, 14),
          transform: `translateY(${rise(frame, 14)}px)`,
        }}
      >
        {chapter.name}
      </div>
      <div
        style={{
          fontSize: 30,
          color: DIM,
          textAlign: "center",
          marginTop: 30,
          maxWidth: 950,
          opacity: fade(frame, 30),
          transform: `translateY(${rise(frame, 30)}px)`,
        }}
      >
        {chapter.narration}
      </div>
    </AbsoluteFill>
  );
};

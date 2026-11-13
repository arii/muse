import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { BG, FONT, GOLD, INK } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";

// Chapter with a heading and a list of bullets that enter one by one.

export const BulletList: React.FC<{ chapter: Chapter }> = ({ chapter }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        backgroundColor: BG,
        fontFamily: FONT,
        padding: "70px 90px",
      }}
    >
      <div
        style={{
          fontSize: 58,
          fontWeight: 800,
          color: GOLD,
          opacity: fade(frame),
          transform: `translateY(${rise(frame)}px)`,
        }}
      >
        {chapter.name}
      </div>
      <div style={{ marginTop: 60, display: "flex", flexDirection: "column", gap: 34 }}>
        {chapter.bullets.map((bullet, i) => {
          const start = 14 + i * 16;
          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 24,
                fontSize: 40,
                color: INK,
                opacity: fade(frame, start),
                transform: `translateY(${rise(frame, start)}px)`,
              }}
            >
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: 999,
                  backgroundColor: GOLD,
                  flexShrink: 0,
                  alignSelf: "center",
                }}
              />
              <div>{bullet}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

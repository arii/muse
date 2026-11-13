import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { BG, DIM, FONT, GOLD, INK, LINE, PANEL } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";
import { Terminal } from "./Terminal";

// Two-column chapter: narration text on the left, bullets or a terminal
// panel on the right (whichever the chapter carries).

export const Split: React.FC<{ chapter: Chapter }> = ({ chapter }) => {
  const frame = useCurrentFrame();
  const showTerminal = chapter.terminal.length > 0;
  return (
    <AbsoluteFill
      style={{
        backgroundColor: BG,
        fontFamily: FONT,
        padding: "70px 90px",
        display: "flex",
        flexDirection: "row",
        gap: 60,
      }}
    >
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          opacity: fade(frame),
          transform: `translateY(${rise(frame)}px)`,
        }}
      >
        <div style={{ fontSize: 52, fontWeight: 800, color: GOLD, lineHeight: 1.2 }}>
          {chapter.name}
        </div>
        <div style={{ fontSize: 32, color: INK, marginTop: 34, lineHeight: 1.45 }}>
          {chapter.narration}
        </div>
      </div>
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          opacity: fade(frame, 12),
          transform: `translateY(${rise(frame, 12)}px)`,
        }}
      >
        {showTerminal ? (
          <Terminal chapter={chapter} compact />
        ) : (
          <div
            style={{
              backgroundColor: PANEL,
              border: `2px solid ${LINE}`,
              borderRadius: 16,
              padding: "36px 40px",
              display: "flex",
              flexDirection: "column",
              gap: 26,
            }}
          >
            {chapter.bullets.map((bullet, i) => (
              <div
                key={i}
                style={{
                  fontSize: 28,
                  color: DIM,
                  opacity: fade(frame, 20 + i * 14),
                }}
              >
                {bullet}
              </div>
            ))}
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};

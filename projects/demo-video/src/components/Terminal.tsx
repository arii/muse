import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { BG, DIM, FONT, GREEN, INK, LINE, MONO, PANEL } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";

// Chapter showing a terminal window whose lines appear one by one.
// Lines from fenced code blocks in the script; a leading "$ " renders as a
// command prompt in green.

export const Terminal: React.FC<{ chapter: Chapter; compact?: boolean }> = ({
  chapter,
  compact = false,
}) => {
  const frame = useCurrentFrame();
  const lineStart = (i: number) => 16 + i * 14;
  return (
    <div
      style={{
        backgroundColor: PANEL,
        border: `2px solid ${LINE}`,
        borderRadius: 16,
        overflow: "hidden",
        width: "100%",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "14px 20px",
          borderBottom: `2px solid ${LINE}`,
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          {["#f87171", "#f5b942", "#4ade80"].map((color) => (
            <div
              key={color}
              style={{ width: 14, height: 14, borderRadius: 999, backgroundColor: color }}
            />
          ))}
        </div>
        <div style={{ fontFamily: MONO, fontSize: 18, color: DIM }}>{chapter.name}</div>
      </div>
      <div
        style={{
          padding: compact ? "22px 26px" : "30px 34px",
          fontFamily: MONO,
          fontSize: compact ? 22 : 26,
          lineHeight: 1.7,
        }}
      >
        {chapter.terminal.map((line, i) => {
          const isCommand = line.startsWith("$ ");
          const opacity = fade(frame, lineStart(i), 12);
          return (
            <div
              key={i}
              style={{
                color: isCommand ? GREEN : INK,
                whiteSpace: "pre-wrap",
                opacity,
                transform: `translateY(${rise(frame, lineStart(i), 12, 10)}px)`,
              }}
            >
              {line}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Full-screen terminal chapter (visual: terminal), narration spoken over it.

export const TerminalChapter: React.FC<{ chapter: Chapter }> = ({ chapter }) => (
  <AbsoluteFill
    style={{
      backgroundColor: BG,
      fontFamily: FONT,
      padding: "70px 90px",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
    }}
  >
    <Terminal chapter={chapter} />
  </AbsoluteFill>
);

import React from "react";

// Shared theme + bits for the grasping-2026 review clips.

export const BG = "#0b0e14";
export const PANEL = "#151b28";
export const INK = "#f2f5fa";
export const DIM = "#9aa6bd";
export const GOLD = "#f5b942";
export const GREEN = "#4ade80";
export const RED = "#f87171";
export const LINE = "#2a3350";

export const FONT =
  "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

export const Badge: React.FC<{ win: boolean; text: string; size?: number }> = ({
  win,
  text,
  size = 34,
}) => (
  <div
    style={{
      display: "inline-block",
      fontFamily: FONT,
      fontSize: size,
      fontWeight: 800,
      color: win ? "#06130b" : "#1c0606",
      backgroundColor: win ? GREEN : RED,
      borderRadius: 999,
      padding: `${size * 0.12}px ${size * 0.5}px`,
      lineHeight: 1.2,
      whiteSpace: "nowrap",
    }}
  >
    {text}
  </div>
);

export const Footer: React.FC<{ text: string }> = ({ text }) => (
  <div
    style={{
      fontFamily: FONT,
      fontSize: 28,
      color: DIM,
      textAlign: "center",
      whiteSpace: "nowrap",
    }}
  >
    {text}
  </div>
);

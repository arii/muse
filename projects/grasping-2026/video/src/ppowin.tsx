import React from "react";
import {
  AbsoluteFill,
  AnimatedImage,
  Easing,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { BG, DIM, FONT, GOLD, INK, LINE, Badge } from "./theme";

const easeOut = Easing.bezier(0.23, 1, 0.32, 1);

// PPO vs heuristic on seed 1 (compare-1). Seed-1 truth is shown honestly:
// the heuristic wins this seed (FC, caged) while PPO misses (shove, 2.19 m
// displacement — compare-report.md). The caption carries the verified totals
// across all 5 scenes × seeds 0–9: PPO pc5 40/50, heuristic 24/50
// (10+0+2+6+6 from the site results matrix).

const PANEL_W = 360;
const PANEL_H = Math.round((PANEL_W * 388) / 480); // 291

const Panel: React.FC<{
  name: string;
  file: string;
  win: boolean;
  badge: string;
  fromLeft: boolean;
}> = ({ name, file, win, badge, fromLeft }) => {
  const frame = useCurrentFrame();
  const dx = interpolate(frame, [10, 40], [fromLeft ? -80 : 80, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const opacity = interpolate(frame, [10, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  return (
    <div
      style={{
        opacity,
        transform: `translateX(${dx}px)`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
      }}
    >
      <div style={{ fontSize: 40, fontWeight: 900, color: INK }}>{name}</div>
      <div
        style={{
          borderRadius: 14,
          border: `3px solid ${LINE}`,
          overflow: "hidden",
          height: PANEL_H,
        }}
      >
        <AnimatedImage
          src={staticFile(file)}
          width={PANEL_W}
          height={PANEL_H}
          fit="fill"
          loopBehavior="loop"
        />
      </div>
      <Badge win={win} text={badge} size={28} />
    </div>
  );
};

export const PpoWin: React.FC = () => {
  const frame = useCurrentFrame();
  const captionOpacity = interpolate(frame, [200, 225], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const captionDy = interpolate(frame, [200, 225], [24, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  return (
    <AbsoluteFill
      style={{
        backgroundColor: BG,
        fontFamily: FONT,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        gap: 5,
        padding: "12px 40px 0",
      }}
    >
      <div style={{ fontSize: 50, fontWeight: 900, color: INK, lineHeight: 1.1 }}>
        PPO vs heuristic
      </div>
      <div style={{ fontSize: 30, color: DIM, lineHeight: 1.1 }}>
        same seed · seed 1 · random scene
      </div>
      <div style={{ display: "flex", gap: 40, marginTop: 2 }}>
        <Panel
          name="PPO"
          file="ppo.gif"
          win={false}
          badge="✗ seed 1: miss"
          fromLeft
        />
        <Panel
          name="Heuristic"
          file="heuristic.gif"
          win
          badge="✓ seed 1: FC"
          fromLeft={false}
        />
      </div>
      <div
        style={{
          opacity: captionOpacity,
          transform: `translateY(${captionDy}px)`,
          marginTop: 0,
          textAlign: "center",
          backgroundColor: "#1a1407",
          border: `3px solid ${GOLD}`,
          borderRadius: 14,
          padding: "4px 24px",
        }}
      >
        <div style={{ fontSize: 26, color: DIM, lineHeight: 1.2 }}>
          Verified totals · 5 scenes × seeds 0–9
        </div>
        <div
          style={{
            fontSize: 33,
            fontWeight: 900,
            color: GOLD,
            whiteSpace: "nowrap",
            lineHeight: 1.2,
          }}
        >
          PPO pc5 40/50 FC · best overall — heuristic 24/50
        </div>
      </div>
    </AbsoluteFill>
  );
};

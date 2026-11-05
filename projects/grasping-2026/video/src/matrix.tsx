import React from "react";
import { AbsoluteFill, Easing, interpolate, useCurrentFrame } from "remotion";
import { BG, DIM, FONT, GOLD, GREEN, INK, LINE, PANEL } from "./theme";

const easeOut = Easing.bezier(0.23, 1, 0.32, 1);

// Animated results matrix. Every number is from the verified scene × planner
// matrix in site/index.html (FC /10, seeds 0–9):
//   PPO pc5:      9/9/8/9/5  = 40
//   BC boosted:   9/9/8/6/6  = 38
//   Autopush:    10/0/2/6/8  = 26
//   Heuristic:   10/0/2/6/6  = 24
//   BC base:      7/2/1/4/4  = 18
//   Diffusion:    5/0/0/5/3  = 13

const SCENES = ["random", "surround", "decoy", "sandwich", "adjacent"];

const ROWS: { name: string; tag?: string; best?: boolean; vals: number[] }[] = [
  { name: "PPO pc5", tag: "RL", best: true, vals: [9, 9, 8, 9, 5] },
  { name: "BC boosted", tag: "clone", vals: [9, 9, 8, 6, 6] },
  { name: "Autopush", tag: "script", vals: [10, 0, 2, 6, 8] },
  { name: "Heuristic", tag: "script", vals: [10, 0, 2, 6, 6] },
  { name: "BC base", tag: "clone", vals: [7, 2, 1, 4, 4] },
  { name: "Diffusion", tag: "clone", vals: [5, 0, 0, 5, 3] },
];

const NAME_W = 270;
const CELL_W = 130;
const TOTAL_W = 112;
const ROW_H = 56;
const HEADER_H = 48;

const cellBg = (v: number) =>
  v >= 8
    ? "rgba(74,222,128,0.30)"
    : v >= 5
      ? "rgba(74,222,128,0.16)"
      : v >= 2
        ? "rgba(245,185,66,0.14)"
        : "rgba(248,113,113,0.14)";

const MatrixRow: React.FC<{ row: (typeof ROWS)[number]; index: number }> = ({
  row,
  index,
}) => {
  const frame = useCurrentFrame();
  const enter = 70 + index * 18;
  const opacity = interpolate(frame, [enter, enter + 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const dx = interpolate(frame, [enter, enter + 20], [-40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const total = row.vals.reduce((a, b) => a + b, 0);
  const shown = Math.round(
    interpolate(frame, [enter + 20, enter + 50], [0, total], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }),
  );
  return (
    <div
      style={{
        opacity,
        transform: `translateX(${dx}px)`,
        display: "flex",
        height: ROW_H,
        alignItems: "stretch",
        backgroundColor: row.best ? "#1a1407" : PANEL,
        border: row.best ? `3px solid ${GOLD}` : `2px solid ${LINE}`,
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: NAME_W,
          display: "flex",
          alignItems: "center",
          gap: 10,
          paddingLeft: 18,
          fontSize: 36,
          fontWeight: row.best ? 900 : 700,
          color: row.best ? GOLD : INK,
          whiteSpace: "nowrap",
        }}
      >
        {row.best ? "★ " : ""}
        {row.name}
        <span style={{ fontSize: 24, fontWeight: 600, color: DIM }}>
          {row.tag}
        </span>
      </div>
      {row.vals.map((v, i) => (
        <div
          key={i}
          style={{
            width: CELL_W,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 36,
            fontWeight: 800,
            color: INK,
            backgroundColor: cellBg(v),
            borderLeft: `2px solid ${BG}`,
          }}
        >
          {v}
        </div>
      ))}
      <div
        style={{
          width: TOTAL_W,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 40,
          fontWeight: 900,
          color: row.best ? GOLD : GREEN,
          borderLeft: `2px solid ${BG}`,
        }}
      >
        {shown}
      </div>
    </div>
  );
};

export const Matrix: React.FC = () => {
  const frame = useCurrentFrame();
  const headOpacity = interpolate(frame, [20, 45], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const headDy = interpolate(frame, [20, 45], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const footOpacity = interpolate(frame, [345, 370], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  const footDy = interpolate(frame, [345, 370], [12, 0], {
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
        gap: 8,
        padding: "56px 0 0",
      }}
    >
      <div
        style={{
          opacity: headOpacity,
          transform: `translateY(${headDy}px)`,
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: 58, fontWeight: 900, color: INK, lineHeight: 1.1 }}>
          Results matrix
        </div>
        <div style={{ fontSize: 32, color: DIM }}>
          FC wins /10 · seeds 0–9 · best overall:{" "}
          <span style={{ color: GOLD, fontWeight: 800 }}>PPO pc5 40/50</span>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <div
          style={{
            display: "flex",
            height: HEADER_H,
            alignItems: "center",
            opacity: headOpacity,
            transform: `translateY(${headDy}px)`,
          }}
        >
          <div style={{ width: NAME_W }} />
          {SCENES.map((s) => (
            <div
              key={s}
              style={{
                width: CELL_W,
                textAlign: "center",
                fontSize: 24,
                fontWeight: 700,
                color: DIM,
                whiteSpace: "nowrap",
              }}
            >
              {s}
            </div>
          ))}
          <div
            style={{
              width: TOTAL_W,
              textAlign: "center",
              fontSize: 30,
              fontWeight: 700,
              color: DIM,
            }}
          >
            total
          </div>
        </div>
        {ROWS.map((r, i) => (
          <MatrixRow key={r.name} row={r} index={i} />
        ))}
      </div>
      <div
        style={{
          opacity: footOpacity,
          transform: `translateY(${footDy}px)`,
          fontSize: 26,
          color: DIM,
        }}
      >
        source: site results matrix · scene × planner FC /10
      </div>
    </AbsoluteFill>
  );
};

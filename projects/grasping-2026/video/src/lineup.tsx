import React from "react";
import {
  AbsoluteFill,
  AnimatedImage,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { BG, DIM, FONT, INK, LINE, Badge, Footer } from "./theme";

// Seed-1 outcomes verified against episodes/compare-0-9.json (FC flags) and
// episodes/compare-report.md (autopush seed 1 FC=True). VLM seed 1: full-run
// 9/10 with lone miss on seed 8 (site/index.html), so seed 1 is a win.
const PLANNERS: {
  name: string;
  file: string;
  win: boolean;
  note: string;
}[] = [
  { name: "Heuristic", file: "heuristic.gif", win: true, note: "FC · caged" },
  { name: "Autopush", file: "autopush.gif", win: true, note: "FC · caged" },
  { name: "BC", file: "bc.gif", win: true, note: "FC" },
  { name: "PPO", file: "ppo.gif", win: false, note: "miss" },
  { name: "Diffusion", file: "diffusion.gif", win: false, note: "miss" },
  { name: "VLM", file: "vlm.gif", win: true, note: "FC" },
  { name: "Reactive", file: "reactive.gif", win: false, note: "miss" },
];

const ROW1 = PLANNERS.slice(0, 4);
const ROW2 = PLANNERS.slice(4);

const TITLE_OUT = 60;
const GRID_START = 78;

const Cell: React.FC<{
  name: string;
  file: string;
  win: boolean;
  note: string;
  index: number;
  cellW: number;
}> = ({ name, file, win, note, index, cellW }) => {
  const frame = useCurrentFrame();
  const enter = GRID_START + index * 14;
  const opacity = interpolate(frame, [enter, enter + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const dy = interpolate(frame, [enter, enter + 18], [36, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const badgeOpacity = interpolate(frame, [330, 350], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const gifH = Math.round((cellW * 388) / 480);
  return (
    <div
      style={{
        width: cellW,
        opacity,
        transform: `translateY(${dy}px)`,
        backgroundColor: "#151b28",
        borderRadius: 14,
        border: `2px solid ${LINE}`,
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div
        style={{
          fontFamily: FONT,
          fontSize: 36,
          fontWeight: 800,
          color: INK,
          textAlign: "center",
          padding: "6px 0 2px",
        }}
      >
        {name}
      </div>
      <AnimatedImage
        src={staticFile(file)}
        width={cellW}
        height={gifH}
        fit="fill"
        loopBehavior="loop"
      />
      <div
        style={{
          position: "absolute",
          right: 10,
          bottom: 10,
          opacity: badgeOpacity,
        }}
      >
        <Badge win={win} text={win ? `✓ ${note}` : `✗ ${note}`} size={26} />
      </div>
    </div>
  );
};

export const Lineup: React.FC = () => {
  const frame = useCurrentFrame();
  const titleOpacity = interpolate(frame, [TITLE_OUT, TITLE_OUT + 15], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const gridOpacity = interpolate(frame, [GRID_START, GRID_START + 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ backgroundColor: BG, fontFamily: FONT }}>
      {frame < TITLE_OUT + 15 && (
        <AbsoluteFill
          style={{
            opacity: titleOpacity,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 18,
          }}
        >
          <div
            style={{
              fontSize: 44,
              fontWeight: 700,
              letterSpacing: 6,
              color: DIM,
            }}
          >
            WHOLE-ARM GRASPING
          </div>
          <div style={{ fontSize: 104, fontWeight: 900, color: INK }}>
            Planner lineup
          </div>
          <div style={{ fontSize: 52, color: DIM }}>
            7 planners · same scene · same seed (seed 1)
          </div>
          <div
            style={{
              fontSize: 44,
              color: INK,
              backgroundColor: "#151b28",
              border: `2px solid ${LINE}`,
              borderRadius: 12,
              padding: "8px 28px",
              marginTop: 10,
            }}
          >
            “grasp the red box in isolation”
          </div>
        </AbsoluteFill>
      )}
      {frame >= GRID_START && (
        <AbsoluteFill
          style={{
            opacity: gridOpacity,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 14,
            padding: "20px 32px",
          }}
        >
          <div
            style={{
              fontSize: 46,
              fontWeight: 800,
              color: INK,
              whiteSpace: "nowrap",
            }}
          >
            Planner lineup · seed 1 · same scene
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            {ROW1.map((p, i) => (
              <Cell key={p.name} {...p} index={i} cellW={270} />
            ))}
          </div>
          <div style={{ display: "flex", gap: 16 }}>
            {ROW2.map((p, i) => (
              <Cell key={p.name} {...p} index={i + 4} cellW={270} />
            ))}
          </div>
          <Footer text="badges = seed-1 force closure · verified: episodes/compare-0-9.json + compare-report.md" />
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};

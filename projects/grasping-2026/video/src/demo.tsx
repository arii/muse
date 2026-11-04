import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { BG, DIM, FONT, GOLD, GREEN, INK, LINE } from "./theme";
import { Lineup } from "./lineup";
import { PpoWin } from "./ppowin";
import { Matrix } from "./matrix";

// 3-minute-track demo video: 3660 frames @30fps ≈ 122s.
// Chapters: title / pipeline / SAM hero / lineup / ppo-win / matrix / close.

const fade = (frame: number, start = 0, len = 24) =>
  interpolate(frame, [start, start + len], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

const rise = (frame: number, start = 0, len = 24, dist = 28) =>
  interpolate(frame, [start, start + len], [dist, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

const Center: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill
    style={{
      backgroundColor: BG,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: FONT,
      padding: 60,
    }}
  >
    {children}
  </AbsoluteFill>
);

// ---------------- Title ----------------
const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Center>
      <div
        style={{
          opacity: fade(frame, 10),
          transform: `translateY(${rise(frame, 10)}px)`,
          fontSize: 84,
          fontWeight: 800,
          color: INK,
          textAlign: "center",
          lineHeight: 1.15,
        }}
      >
        Grasp the red box
        <br />
        in isolation.
      </div>
      <div
        style={{
          opacity: fade(frame, 45),
          marginTop: 28,
          fontSize: 40,
          color: DIM,
          textAlign: "center",
        }}
      >
        Language-conditioned whole-arm grasping, in simulation.
      </div>
      <div
        style={{
          opacity: fade(frame, 80),
          marginTop: 44,
          fontSize: 30,
          color: GOLD,
          fontWeight: 700,
        }}
      >
        Perception by SAM 3.1 · Meta Model API
      </div>
    </Center>
  );
};

// ---------------- Pipeline ----------------
const STAGES_ROW1 = [
  { name: "Instruction", desc: "words in" },
  { name: "VLM grounding", desc: "what + where" },
  { name: "SAM 3.1", desc: "words → pixels", hot: true },
];
const STAGES_ROW2 = [
  { name: "Grasp synthesis", desc: "how to wrap" },
  { name: "Motion planning", desc: "joint targets" },
  { name: "MuJoCo", desc: "execution" },
];

const StageBox: React.FC<{ name: string; desc: string; hot?: boolean; on: number }> = ({
  name,
  desc,
  hot,
  on,
}) => (
  <div
    style={{
      width: 300,
      backgroundColor: "#151b28",
      border: `3px solid ${hot ? GOLD : LINE}`,
      borderRadius: 16,
      padding: "16px 12px",
      textAlign: "center",
      opacity: 0.25 + 0.75 * on,
      transform: `translateY(${(1 - on) * 24}px)`,
      boxShadow: hot ? `0 0 32px ${GOLD}55` : "none",
      flexShrink: 0,
    }}
  >
    <div style={{ fontSize: 34, fontWeight: 800, color: INK, lineHeight: 1.2 }}>
      {name}
    </div>
    <div style={{ fontSize: 26, color: DIM, marginTop: 8 }}>{desc}</div>
  </div>
);

const Arrow: React.FC<{ on: number; dir?: "right" | "left" | "down" }> = ({
  on,
  dir = "right",
}) => (
  <div
    style={{
      fontSize: 44,
      color: DIM,
      alignSelf: "center",
      opacity: on,
      flexShrink: 0,
      width: 44,
      textAlign: "center",
    }}
  >
    {dir === "right" ? "→" : dir === "left" ? "←" : "↓"}
  </div>
);

const Pipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const on = (start: number) =>
    interpolate(frame, [start, start + 30], [0, 1], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  return (
    <Center>
      <div
        style={{
          opacity: fade(frame, 0),
          fontSize: 52,
          fontWeight: 800,
          color: INK,
          marginBottom: 40,
        }}
      >
        One instruction, six stages
      </div>
      {/* row 1: left to right */}
      <div style={{ display: "flex", alignItems: "center" }}>
        {STAGES_ROW1.map((s, i) => (
          <React.Fragment key={s.name}>
            <StageBox name={s.name} desc={s.desc} hot={s.hot} on={on(30 + i * 60)} />
            {i < STAGES_ROW1.length - 1 && <Arrow on={on(60 + i * 60)} dir="right" />}
          </React.Fragment>
        ))}
      </div>
      {/* down connector on the right */}
      <div
        style={{
          width: 300,
          marginLeft: 300 + 44 + 300 + 44,
          textAlign: "center",
          opacity: on(210),
        }}
      >
        <div style={{ fontSize: 44, color: DIM }}>↓</div>
      </div>
      {/* row 2: right to left (snake) */}
      <div style={{ display: "flex", alignItems: "center" }}>
        {[...STAGES_ROW2].reverse().map((s, i) => (
          <React.Fragment key={s.name}>
            <StageBox name={s.name} desc={s.desc} on={on(240 + i * 60)} />
            {i < STAGES_ROW2.length - 1 && <Arrow on={on(270 + i * 60)} dir="left" />}
          </React.Fragment>
        ))}
      </div>
      <div
        style={{
          opacity: fade(frame, 430),
          marginTop: 36,
          fontSize: 30,
          color: DIM,
          textAlign: "center",
          maxWidth: 1000,
        }}
      >
        The VLM reads the camera image and plans. SAM 3.1 turns its words
        into a mask. Geometry turns the mask into motion.
      </div>
    </Center>
  );
};

// ---------------- SAM hero ----------------
const SamHero: React.FC = () => {
  const frame = useCurrentFrame();
  // sub-timeline (relative): prompt 0-300, reveal 300-750, split 750-1350
  const promptText = '"red box"';
  const typed = Math.min(
    promptText.length,
    Math.floor(interpolate(frame, [60, 200], [0, promptText.length], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }))
  );
  const reveal = interpolate(frame, [330, 700], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const mainOut = interpolate(frame, [740, 810], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const split = interpolate(frame, [820, 900], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const imgW = 880;
  const imgH = Math.round((imgW * 480) / 640);

  return (
    <AbsoluteFill style={{ backgroundColor: BG, fontFamily: FONT }}>
      <div
        style={{
          position: "absolute",
          top: 44,
          width: "100%",
          textAlign: "center",
          fontSize: 52,
          fontWeight: 800,
          color: INK,
          opacity: fade(frame, 0),
        }}
      >
        SAM 3.1 sees the robot's camera
      </div>

      {/* prompt line */}
      <div
        style={{
          position: "absolute",
          top: 130,
          width: "100%",
          textAlign: "center",
          fontSize: 40,
          color: DIM,
          opacity: fade(frame, 30),
        }}
      >
        prompt:{" "}
        <span style={{ color: GOLD, fontWeight: 700, fontFamily: "monospace" }}>
          {promptText.slice(0, typed)}
        </span>
        <span style={{ color: GOLD }}>▍</span>
      </div>

      {/* main stage: frame with wipe-revealed overlay */}
      <div
        style={{
          position: "absolute",
          top: 210,
          left: (1280 - imgW) / 2,
          width: imgW,
          height: imgH,
          opacity: mainOut,
        }}
      >
        <Img
          src={staticFile("sam_frame.png")}
          style={{ position: "absolute", width: imgW, height: imgH, borderRadius: 12 }}
        />
        <div
          style={{
            position: "absolute",
            width: imgW * reveal,
            height: imgH,
            overflow: "hidden",
            borderRadius: 12,
          }}
        >
          <Img src={staticFile("sam_overlay.png")} style={{ width: imgW, height: imgH }} />
        </div>
        <div
          style={{
            position: "absolute",
            bottom: 12,
            left: 12,
            fontSize: 26,
            color: INK,
            backgroundColor: "#000000aa",
            padding: "6px 14px",
            borderRadius: 8,
            opacity: interpolate(frame, [700, 760], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          Live output · Meta Model API · model sam-3.1
        </div>
      </div>

      {/* split view: frame | overlay | mask */}
      <div
        style={{
          position: "absolute",
          top: 210,
          width: "100%",
          display: split > 0 ? "flex" : "none",
          justifyContent: "center",
          gap: 24,
          opacity: split,
        }}
      >
        {[
          { src: "sam_frame.png", label: "camera" },
          { src: "sam_overlay.png", label: "mask + box" },
          { src: "sam_mask.png", label: "mask" },
        ].map((p) => (
          <div key={p.src} style={{ textAlign: "center" }}>
            <Img
              src={staticFile(p.src)}
              style={{ width: 380, height: 285, borderRadius: 12, border: `2px solid ${LINE}` }}
            />
            <div style={{ fontSize: 28, color: DIM, marginTop: 10 }}>{p.label}</div>
          </div>
        ))}
      </div>

      {/* honest footer */}
      <div
        style={{
          position: "absolute",
          bottom: 36,
          width: "100%",
          textAlign: "center",
          fontSize: 28,
          color: DIM,
          opacity: interpolate(frame, [950, 1030], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Checked against simulator ground truth (IoU 0.77) — ground truth never
        produced it. The model saw only the image.
        <br />
        <span style={{ color: GOLD }}>
          Perception stage verified live · closed-loop integration next
        </span>
      </div>
    </AbsoluteFill>
  );
};

// ---------------- Close ----------------
const Close: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <Center>
      <div
        style={{
          opacity: fade(frame, 10),
          fontSize: 72,
          fontWeight: 800,
          color: INK,
          textAlign: "center",
          lineHeight: 1.2,
        }}
      >
        A robot arm that
        <br />
        grasps what you name.
      </div>
      <div
        style={{
          opacity: fade(frame, 60),
          marginTop: 40,
          fontSize: 34,
          color: DIM,
          textAlign: "center",
          lineHeight: 1.7,
        }}
      >
        github.com/arii/muse
        <br />
        Built with SAM 3.1 on the Meta Model API
      </div>
      <div
        style={{
          opacity: fade(frame, 110),
          marginTop: 24,
          display: "flex",
          gap: 16,
        }}
      >
        {["Indie track", "Push the Frontier"].map((t) => (
          <div
            key={t}
            style={{
              fontSize: 28,
              fontWeight: 700,
              color: "#06130b",
              backgroundColor: GREEN,
              borderRadius: 999,
              padding: "8px 26px",
            }}
          >
            {t}
          </div>
        ))}
      </div>
    </Center>
  );
};

// ---------------- Assembly ----------------
export const Demo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: BG }}>
      <Sequence from={0} durationInFrames={300}>
        <Audio src={staticFile("vo/title.mp3")} />
        <TitleCard />
      </Sequence>
      <Sequence from={300} durationInFrames={450}>
        <Audio src={staticFile("vo/pipeline.mp3")} />
        <Pipeline />
      </Sequence>
      <Sequence from={750} durationInFrames={1350}>
        <Audio src={staticFile("vo/sam.mp3")} />
        <SamHero />
      </Sequence>
      <Sequence from={2100} durationInFrames={450}>
        <Audio src={staticFile("vo/lineup.mp3")} />
        <Lineup />
      </Sequence>
      <Sequence from={2550} durationInFrames={360}>
        <Audio src={staticFile("vo/ppowin.mp3")} />
        <PpoWin />
      </Sequence>
      <Sequence from={2910} durationInFrames={450}>
        <Audio src={staticFile("vo/matrix.mp3")} />
        <Matrix />
      </Sequence>
      <Sequence from={3360} durationInFrames={300}>
        <Audio src={staticFile("vo/close.mp3")} />
        <Close />
      </Sequence>
    </AbsoluteFill>
  );
};

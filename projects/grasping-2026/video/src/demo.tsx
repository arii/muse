import React from "react";
import {
  AbsoluteFill,
  Easing,
  Img,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { BG, DIM, FONT, GOLD, GREEN, INK, LINE } from "./theme";
import { Captions } from "./captions";
import { Lineup } from "./lineup";
import { PpoWin } from "./ppowin";
import { Matrix } from "./matrix";

// 3-minute-track demo video: 3660 frames @30fps ≈ 122s.
// Chapters: title / pipeline / SAM hero / lineup / ppo-win / matrix / close.

// Strong ease-out on all entrances (linear reads mechanical).
const easeOut = Easing.bezier(0.23, 1, 0.32, 1);

const fade = (frame: number, start = 0, len = 24) =>
  interpolate(frame, [start, start + len], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });

const rise = (frame: number, start = 0, len = 24, dist = 28) =>
  interpolate(frame, [start, start + len], [dist, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
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
          transform: `translateY(${rise(frame, 45)}px)`,
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
          transform: `translateY(${rise(frame, 80)}px)`,
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
      easing: easeOut,
    });
  return (
    <Center>
      <div
        style={{
          opacity: fade(frame, 0),
          transform: `translateY(${rise(frame, 0)}px)`,
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
          transform: `translateY(${rise(frame, 430, 20, 16)}px)`,
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
  const promptText = "\u201cred box\u201d";
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
    easing: easeOut,
  });
  const mainOut = interpolate(frame, [740, 810], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const split = interpolate(frame, [820, 900], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easeOut,
  });
  // Full-frame stills (2026-11-04): the old 320x240 center crop showed the
  // arm base as a close-up with links running off-frame. Full frame keeps
  // the whole arm and all boxes visible; a slow push-in keeps the stills
  // in motion. Stage geometry leaves the bottom ~90px to the caption band.
  const imgW = 624;
  const imgH = Math.round((imgW * 480) / 640);

  return (
    <AbsoluteFill style={{ backgroundColor: BG, fontFamily: FONT }}>
      <div
        style={{
          position: "absolute",
          top: 28,
          width: "100%",
          textAlign: "center",
          fontSize: 48,
          fontWeight: 800,
          color: INK,
          opacity: fade(frame, 0),
          transform: `translateY(${rise(frame, 0, 24, 20)}px)`,
        }}
      >
        SAM 3.1 sees the robot's camera
      </div>

      {/* prompt line */}
      <div
        style={{
          position: "absolute",
          top: 100,
          width: "100%",
          textAlign: "center",
          fontSize: 36,
          color: DIM,
          opacity: fade(frame, 30),
          transform: `translateY(${rise(frame, 30, 24, 16)}px)`,
        }}
      >
        prompt:{" "}
        <span style={{ color: GOLD, fontWeight: 700 }}>
          {promptText.slice(0, typed)}
        </span>
        <span style={{ color: GOLD }}>▍</span>
      </div>

      {/* main stage: frame with wipe-revealed overlay */}
      <div
        style={{
          position: "absolute",
          top: 160,
          left: (1280 - imgW) / 2,
          width: imgW,
          height: imgH,
          opacity: mainOut,
          scale: interpolate(frame, [0, 810], [1, 1.045], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <Img
          src={staticFile("sam_frame.png")}
          style={{ position: "absolute", width: imgW, height: imgH, borderRadius: 12 }}
        />
        <Img
          src={staticFile("sam_overlay.png")}
          style={{
            position: "absolute",
            width: imgW,
            height: imgH,
            borderRadius: 12,
            clipPath: `inset(0 ${(1 - reveal) * 100}% 0 0 round 12px)`,
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: 12,
            left: 12,
            fontSize: 24,
            color: INK,
            backgroundColor: "#000000aa",
            padding: "6px 14px",
            borderRadius: 8,
            lineHeight: 1.4,
            opacity: interpolate(frame, [700, 760], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          Live output · Meta Model API · sam-3.1 · IoU 0.77
          <br />
          <span style={{ color: GOLD }}>
            Perception verified live · closed-loop integration next
          </span>
        </div>
      </div>

      {/* split view: frame | overlay | mask */}
      <div
        style={{
          position: "absolute",
          top: 190,
          width: "100%",
          display: split > 0 ? "flex" : "none",
          justifyContent: "center",
          gap: 24,
          opacity: split,
          scale: interpolate(frame, [820, 1350], [1, 1.04], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
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
              style={{ width: 400, height: 300, borderRadius: 12, border: `2px solid ${LINE}` }}
            />
            <div style={{ fontSize: 28, color: DIM, marginTop: 10 }}>{p.label}</div>
          </div>
        ))}
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
          transform: `translateY(${rise(frame, 60)}px)`,
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
          transform: `translateY(${rise(frame, 110, 24, 20)}px)`,
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
      <div
        style={{
          opacity: fade(frame, 150),
          transform: `translateY(${rise(frame, 150, 24, 16)}px)`,
          marginTop: 28,
          fontSize: 30,
          fontWeight: 600,
          color: INK,
          textAlign: "center",
        }}
      >
        Made by Ariel Anders and Muse
      </div>
    </Center>
  );
};

// ---------------- Assembly ----------------
// Burned-in captions replace the TTS voiceover (2026-11-04): no <Audio>.
// Cue text is verbatim from VO_SCRIPT.md (now the caption script);
// timings are chapter-relative ms, one cue array per chapter Sequence.
export const Demo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: BG }}>
      <Sequence from={0} durationInFrames={300}>
        <TitleCard />
        <Captions
          cues={[
            {
              text: "Tell a three-jointed arm what to grasp, in words.",
              startMs: 400,
              endMs: 5100,
            },
            { text: "It figures out the rest.", startMs: 5200, endMs: 9800 },
          ]}
        />
      </Sequence>
      <Sequence from={300} durationInFrames={450}>
        <Pipeline />
        <Captions
          cues={[
            { text: "Six stages turn language into motion.", startMs: 200, endMs: 3100 },
            { text: "A vision model grounds the instruction.", startMs: 3200, endMs: 6100 },
            { text: "SAM three point one segments the target.", startMs: 6200, endMs: 9100 },
            { text: "A sampler proposes grasps.", startMs: 9200, endMs: 11900 },
            { text: "And MuJoCo executes.", startMs: 12000, endMs: 14800 },
          ]}
        />
      </Sequence>
      <Sequence from={750} durationInFrames={1350}>
        <SamHero />
        <Captions
          cues={[
            { text: "Here is the perception stage, live.", startMs: 400, endMs: 5900 },
            { text: "The prompt is red box.", startMs: 6000, endMs: 11400 },
            {
              text: "Watch SAM three point one find it among the clutter",
              startMs: 11500,
              endMs: 14700,
            },
            { text: "and trace its exact outline.", startMs: 14800, endMs: 17000 },
            {
              text: "We checked the mask against simulator ground truth:",
              startMs: 17100,
              endMs: 20300,
            },
            {
              text: "seventy-seven percent overlap.",
              startMs: 20400,
              endMs: 22700,
            },
            {
              text: "Ground truth never produced it. The model saw only the image.",
              startMs: 22800,
              endMs: 28300,
            },
            {
              text: "This mask is what the grasp sampler reasons over.",
              startMs: 28400,
              endMs: 33800,
            },
            {
              text: "Boxes touch and overlap in clutter. Bounding boxes go ambiguous.",
              startMs: 33900,
              endMs: 37600,
            },
            { text: "Pixel masks do not.", startMs: 37700, endMs: 39400 },
            {
              text: "That is why segmentation is load-bearing for this task.",
              startMs: 39500,
              endMs: 44800,
            },
          ]}
        />
      </Sequence>
      <Sequence from={2100} durationInFrames={450}>
        <Lineup />
        <Captions
          cues={[
            { text: "Seven planners attempt the same scenes.", startMs: 2200, endMs: 7600 },
            {
              text: "Scripts, reinforcement learning, cloning,",
              startMs: 7700,
              endMs: 11200,
            },
            {
              text: "and vision-language planning, side by side.",
              startMs: 11300,
              endMs: 14800,
            },
          ]}
        />
      </Sequence>
      <Sequence from={2550} durationInFrames={360}>
        <PpoWin />
        <Captions
          cues={[
            {
              text: "The PPO policy wins forty of fifty episodes, the best overall,",
              startMs: 300,
              endMs: 6100,
            },
            {
              text: "by learning restraint: approach gently, disturb nothing.",
              startMs: 6200,
              endMs: 11800,
            },
          ]}
        />
      </Sequence>
      <Sequence from={2910} durationInFrames={450}>
        <Matrix />
        <Captions
          cues={[
            {
              text: "Across five scene types, learned policies match or beat every script.",
              startMs: 300,
              endMs: 7600,
            },
            {
              text: "Cloning at double data ties its own teacher, forty to forty.",
              startMs: 7700,
              endMs: 14800,
            },
          ]}
        />
      </Sequence>
      <Sequence from={3360} durationInFrames={300}>
        <Close />
        <Captions
          cues={[
            { text: "A robot arm that grasps what you name.", startMs: 300, endMs: 5100 },
            {
              text: "Built with SAM three point one on the Meta Model API.",
              startMs: 5200,
              endMs: 9800,
            },
          ]}
        />
      </Sequence>
    </AbsoluteFill>
  );
};

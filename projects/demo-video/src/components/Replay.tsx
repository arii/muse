import React from "react";
import {
  AbsoluteFill,
  OffthreadVideo,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { FONT, GOLD } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";
import { fitReplaySegments, parseReplay } from "../replay";
import { msToFrames } from "../timing";

// Chapter built from recorded live footage of the featured product running —
// e.g. the user's own screen recording of the dashboard. The fence block in
// the script names the clip file under public/replay/<videoId>/. The clip
// plays muted so its ambient audio never competes with narration; a 16:9
// source scales to the full 1280×720 frame with no crop. Cut the asset so it
// covers the chapter's narration; a missing file fails the render explicitly
// (Remotion), never silently.

export const replayClipFile = (videoId: string, spec: string): string =>
  `replay/${videoId}/${spec}`;

export const Replay: React.FC<{ chapter: Chapter; videoId: string }> = ({
  chapter,
  videoId,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const spec = parseReplay(chapter.terminal);
  const segments = fitReplaySegments(spec.segments, msToFrames(chapter.durationMs), fps);
  const src = staticFile(replayClipFile(videoId, spec.file));
  return (
    <AbsoluteFill style={{ backgroundColor: "black", fontFamily: FONT }}>
      {segments.length ? segments.map((segment, i) => (
        <Sequence key={i} from={segment.from} durationInFrames={segment.durationInFrames}>
          <OffthreadVideo
            src={src}
            trimBefore={Math.round(segment.start * fps)}
            trimAfter={Math.round(segment.end * fps)}
            playbackRate={segment.playbackRate}
            muted
            style={{ width: "100%", height: "calc(100% - 120px)", objectFit: "contain" }}
          />
        </Sequence>
      )) : <OffthreadVideo
        src={src}
        muted
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />}
      {!segments.length && <div
        style={{
          position: "absolute",
          top: 44,
          left: 60,
          padding: "10px 22px",
          borderRadius: 10,
          backgroundColor: "rgba(10, 10, 12, 0.72)",
          border: "1px solid rgba(255, 255, 255, 0.18)",
          fontSize: 34,
          fontWeight: 800,
          color: GOLD,
          opacity: fade(frame),
          transform: `translateY(${rise(frame)}px)`,
        }}
      >
        {chapter.name}
      </div>}
      {/* subtle edge vignette so white dashboard chrome meets the caption bar cleanly */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0) 18%, rgba(0,0,0,0) 78%, rgba(0,0,0,0.35) 100%)",
          opacity: interpolate(frame, [0, 24], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      />
    </AbsoluteFill>
  );
};

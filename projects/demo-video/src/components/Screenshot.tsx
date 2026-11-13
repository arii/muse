import React from "react";
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion";
import { BG, FONT, GOLD, LINE, PANEL } from "../theme";
import { fade, rise } from "../motion";
import type { Chapter } from "../script-types";

// Chapter showing a screenshot of the featured repo in a framed panel, with a
// slow zoom so the still frame stays alive under the narration. The image is
// conventionally located at public/screenshots/<videoId>/<chapter-slug>.png —
// a missing file fails the render explicitly (Remotion <Img>), never silently.

export const screenshotFile = (videoId: string, slug: string): string =>
  `screenshots/${videoId}/${slug}.png`;

export const Screenshot: React.FC<{ chapter: Chapter; videoId: string }> = ({
  chapter,
  videoId,
}) => {
  const frame = useCurrentFrame();
  const scale = interpolate(frame, [0, 300], [1.0, 1.06], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill
      style={{
        backgroundColor: BG,
        fontFamily: FONT,
        padding: "60px 90px",
        display: "flex",
        flexDirection: "column",
        gap: 34,
      }}
    >
      <div
        style={{
          fontSize: 44,
          fontWeight: 800,
          color: GOLD,
          opacity: fade(frame),
          transform: `translateY(${rise(frame)}px)`,
        }}
      >
        {chapter.name}
      </div>
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: fade(frame, 12),
          transform: `translateY(${rise(frame, 12)}px)`,
          minHeight: 0,
        }}
      >
        <div
          style={{
            backgroundColor: PANEL,
            border: `2px solid ${LINE}`,
            borderRadius: 16,
            padding: 14,
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            overflow: "hidden",
          }}
        >
          <Img
            src={staticFile(screenshotFile(videoId, chapter.slug))}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              objectFit: "contain",
              borderRadius: 8,
              transform: `scale(${scale})`,
            }}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};

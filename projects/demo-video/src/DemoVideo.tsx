import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { BG } from "./theme";
import { Captions } from "./Captions";
import { Title } from "./components/Title";
import { BulletList } from "./components/BulletList";
import { TerminalChapter } from "./components/Terminal";
import { Split } from "./components/Split";
import type { Chapter, VideoScript } from "./script-types";

// Maps a compiled script plan onto chapter Sequences. Chapter durations come
// from the plan (audio-derived once voiceover exists, silent estimates
// otherwise), so trimming or reordering chapters in the script is enough.

const ChapterCard: React.FC<{ chapter: Chapter; kicker: string }> = ({
  chapter,
  kicker,
}) => {
  switch (chapter.visual) {
    case "bullets":
      return <BulletList chapter={chapter} />;
    case "terminal":
      return <TerminalChapter chapter={chapter} />;
    case "split":
      return <Split chapter={chapter} />;
    case "title":
      return <Title chapter={chapter} kicker={kicker} />;
  }
};

export const DemoVideo: React.FC<{ script: VideoScript }> = ({ script }) => {
  let startMs = 0;
  return (
    <AbsoluteFill style={{ backgroundColor: BG }}>
      {script.chapters.map((chapter) => {
        const from = startMs;
        startMs += chapter.durationMs;
        return (
          <Sequence
            key={chapter.slug}
            from={Math.round((from / 1000) * 30)}
            durationInFrames={Math.max(1, Math.round((chapter.durationMs / 1000) * 30))}
          >
            <ChapterCard chapter={chapter} kicker={script.title} />
            <Captions cues={chapter.cues} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

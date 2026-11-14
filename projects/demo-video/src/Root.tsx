import React from "react";
import {
  Composition,
  type CalculateMetadataFunction,
} from "remotion";
import { DemoVideo, type DemoVideoProps } from "./DemoVideo";
import { getAudioDuration } from "./get-audio-duration";
import {
  chapterVoiceoverFile,
  fitChapterToAudio,
  msToFrames,
} from "./timing";
import type { VideoScript } from "./script-types";

// Compositions are discovered dynamically from src/generated/*.json, one per
// narration script (compiled by `npm run plan`). Adding a video means adding
// a script and running plan — no Root edits, so parallel video PRs never
// conflict here.
//
// import.meta.webpackContext is rspack's native context API (bundler is
// rspack, via remotion.config.ts).

const context = import.meta.webpackContext<VideoScript>("./generated", {
  recursive: false,
  regExp: /\.json$/,
});
const scripts = context
  .keys()
  .sort()
  .map((key) => context(key));

// Voiceover.md pattern: measure each chapter's narration audio and size the
// composition (and pass the audio-fitted chapters via props) so the video is
// exactly as long as its voice. Chapters without audio (silent cards, or
// narration not yet synthesized) keep the plan's word-count estimate.
const calculateDemoMetadata: CalculateMetadataFunction<DemoVideoProps> =
  async ({ props }) => {
    const chapters = await Promise.all(
      props.script.chapters.map(async (chapter) => {
        if (!chapter.narration) return chapter;
        const seconds = await getAudioDuration(
          chapterVoiceoverFile(props.script.id, chapter.slug)
        );
        return fitChapterToAudio(chapter, seconds);
      })
    );
    const script: VideoScript = { ...props.script, chapters };
    return {
      durationInFrames: chapters.reduce(
        (frames, chapter) => frames + msToFrames(chapter.durationMs),
        0
      ),
      props: { script },
    };
  };

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {scripts.map((script) => (
        <Composition
          key={script.id}
          id={script.id}
          component={DemoVideo}
          defaultProps={{ script }}
          calculateMetadata={calculateDemoMetadata}
          width={1280}
          height={720}
          fps={30}
          durationInFrames={Math.max(
            1,
            Math.round((script.durationMs / 1000) * 30)
          )}
        />
      ))}
    </>
  );
};

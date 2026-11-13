import React from "react";
import { Composition } from "remotion";
import { DemoVideo } from "./DemoVideo";
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

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {scripts.map((script) => (
        <Composition
          key={script.id}
          id={script.id}
          component={DemoVideo}
          defaultProps={{ script }}
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

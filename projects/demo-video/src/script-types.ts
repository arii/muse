// The narration-script data model. scripts/plan.ts compiles a markdown
// script into this shape; src/Root.tsx consumes the generated JSON files
// (src/generated/<id>.json). One script drives narration, captions, and
// scene timing, so they cannot drift.

export type Visual = "title" | "bullets" | "terminal" | "split" | "screenshot";

// Cue shape matches @remotion/captions `Caption`
// (text/startMs/endMs, chapter-relative milliseconds).
export type Cue = {
  text: string;
  startMs: number;
  endMs: number;
};

export type Chapter = {
  name: string;
  slug: string;
  visual: Visual;
  // Narration text with numbers spelled as read — the verbatim TTS input.
  narration: string;
  bullets: string[];
  terminal: string[];
  cues: Cue[];
  durationMs: number;
};

export type VideoScript = {
  id: string;
  title: string;
  chapters: Chapter[];
  durationMs: number;
};

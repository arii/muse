// Pure timing helpers shared by scripts/plan.ts (Node side) and src/Root.tsx
// (bundled side). No Node builtins — both contexts import this module.

import type { Chapter, Cue } from "./script-types";

// Matches the 1280×720@30 composition registration.
export const FPS = 30;

export function msToFrames(ms: number): number {
  return Math.max(1, Math.round((ms / 1000) * FPS));
}

export function framesToMs(frames: number): number {
  return (frames / FPS) * 1000;
}

// A narration paragraph becomes one cue per sentence, so caption text tracks
// the voice at sentence granularity. Splits only on whitespace after
// sentence-ending punctuation: a decimal point ("3.5") has no space after it,
// so numbers are never split — and splitting can only separate, never drop
// characters, so cue text stays verbatim against the narration.
export function splitSentences(text: string): string[] {
  const trimmed = text.trim();
  if (trimmed === "") {
    return [];
  }
  return trimmed.split(/(?<=[.!?])\s+/);
}

// Distribute the duration across cues proportionally to word count, so
// captions track the spoken pace. The last cue is pinned to the exact end so
// rounding cannot leave a trailing gap.
export function deriveCues(units: string[], durationMs: number): Cue[] {
  if (units.length === 0) return [];
  const weights = units.map((u) => Math.max(1, countWords(u)));
  const total = weights.reduce((a, b) => a + b, 0);
  let cursor = 0;
  return units.map((text, i) => {
    const span = Math.round((weights[i]! / total) * durationMs);
    const cue: Cue = {
      text,
      startMs: cursor,
      endMs: i === units.length - 1 ? durationMs : cursor + span,
    };
    cursor += span;
    return cue;
  });
}

export function countWords(text: string): number {
  const words = text.trim().match(/\S+/g);
  return words ? words.length : 0;
}

// Scales existing cue timings to a new chapter duration (proportional, still
// anchored at 0 and pinned to the end). Used when measured narration audio
// replaces the word-count estimate.
export function fitCuesToDuration(cues: Cue[], durationMs: number): Cue[] {
  if (cues.length === 0) return [];
  const oldDuration = cues[cues.length - 1]!.endMs;
  if (oldDuration <= 0) return deriveCues(cues.map((c) => c.text), durationMs);
  const scale = durationMs / oldDuration;
  return cues.map((cue, i) => ({
    text: cue.text,
    startMs: Math.round(cue.startMs * scale),
    endMs:
      i === cues.length - 1
        ? durationMs
        : Math.round(cue.endMs * scale),
  }));
}

// Where a chapter's narration audio lives, relative to public/ (staticFile()
// form; the TTS layer writes to the same layout on disk).
export function chapterVoiceoverFile(scriptId: string, slug: string): string {
  return `voiceover/${scriptId}/${slug}.wav`;
}

// Re-times one chapter against its measured narration audio. null (no audio
// file — silent card, or not synthesized yet) keeps the plan estimate. The
// duration is frame-aligned so chapter Sequences and the composition total
// stay in exact frame steps, and cues are re-derived from the same narration
// so text and voice cannot drift.
export function fitChapterToAudio(
  chapter: Chapter,
  audioDurationSec: number | null
): Chapter {
  if (audioDurationSec === null || !chapter.narration) {
    return chapter;
  }
  const frames = msToFrames(Math.round(audioDurationSec * 1000));
  const durationMs = framesToMs(frames);
  return {
    ...chapter,
    durationMs,
    cues: deriveCues(splitSentences(chapter.narration), durationMs),
  };
}

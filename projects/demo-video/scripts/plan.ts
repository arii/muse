// Markdown narration script → compiled VideoScript plan.
//
// Format (see templates/demo-video.md):
//   --- front matter with `id:` (required) and optional `title:` ---
//   # optional display title (used when no front-matter title)
//   ## Chapter name
//   <!-- visual: title|bullets|terminal|split -->   (optional hint)
//   - bullet lines            → shown by bullets/split visuals
//   ``` fenced blocks         → terminal lines
//   plain paragraphs          → narration (TTS input + captions, verbatim
//                               after numbers are spelled as read)
//
// A chapter with no narration text renders as a timed silent card — explicit,
// not accidental (CHAPTER_MIN_MS).

import type { Chapter, Visual, VideoScript } from "../src/script-types";
import { deriveCues, splitSentences } from "../src/timing";
import { spellAsRead } from "./numbers";
import {
  CHAPTER_MIN_MS,
  CHAPTER_PAD_MS,
  WORDS_PER_SECOND,
} from "./config";

const VISUALS: Visual[] = ["title", "bullets", "terminal", "split", "screenshot"];

export class PlanError extends Error {}

const FRONT_MATTER = /^---\n([\s\S]*?)\n---\n/;
const H1 = /^#\s+(.+)$/;
const H2 = /^##\s+(.+)$/;
const VISUAL_HINT = /<!--\s*visual:\s*(\w+)\s*-->/;
const FENCE = /^```/;

export function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function parseFrontMatter(markdown: string): {
  fields: Record<string, string>;
  body: string;
} {
  const match = markdown.match(FRONT_MATTER);
  if (!match) {
    return { fields: {}, body: markdown };
  }
  const fields: Record<string, string> = {};
  for (const line of match[1]!.split("\n")) {
    const kv = line.match(/^(\w+):\s*(.+)$/);
    if (kv) {
      fields[kv[1]!] = kv[2]!.trim();
    }
  }
  return { fields, body: markdown.slice(match[0].length) };
}

export function estimateChapterMs(chapter: {
  narration: string;
}): number {
  const words = chapter.narration.split(/\s+/).filter(Boolean).length;
  if (words === 0) {
    // Explicit silent card: a chapter with no narration is timed, not empty.
    return CHAPTER_MIN_MS;
  }
  return Math.max(CHAPTER_MIN_MS, Math.round((words / WORDS_PER_SECOND) * 1000) + CHAPTER_PAD_MS);
}

// Re-exported so the plan module stays the one-stop import for script
// compilation (tests and callers import deriveCues from here).
export { deriveCues } from "../src/timing";

function parseChapter(header: string, lines: string[]): Chapter {
  const hint = lines.map((l) => l.match(VISUAL_HINT)?.[1]).find(Boolean);
  const visual = (hint ?? "title") as Visual;
  if (!VISUALS.includes(visual)) {
    throw new PlanError(
      `Chapter "${header}": unknown visual "${visual}" (known: ${VISUALS.join(", ")})`
    );
  }

  const bullets: string[] = [];
  const terminal: string[] = [];
  const paragraphs: string[] = [];
  let inFence = false;
  let buffer: string[] = [];

  const flush = () => {
    if (buffer.length > 0) {
      paragraphs.push(buffer.join(" "));
      buffer = [];
    }
  };

  for (const line of lines) {
    if (FENCE.test(line)) {
      if (inFence) {
        flush();
      }
      inFence = !inFence;
      continue;
    }
    if (inFence) {
      terminal.push(line.trim());
      continue;
    }
    if (VISUAL_HINT.test(line)) {
      continue;
    }
    const trimmed = line.trim();
    if (trimmed === "") {
      flush();
      continue;
    }
    if (trimmed.startsWith("- ")) {
      flush();
      bullets.push(trimmed.slice(2).trim());
      continue;
    }
    buffer.push(trimmed);
  }
  flush();

  // Numbers spelled as read: the narration string is the verbatim TTS input
  // and the caption text is cut from it, so they cannot drift. Cues are
  // sentence-level, word-weighted, so captions track the spoken pace.
  const narration = spellAsRead(paragraphs.join(" "));
  const durationMs = estimateChapterMs({ narration });
  const sentences = paragraphs.flatMap((p) => splitSentences(p).map(spellAsRead));
  return {
    name: header,
    slug: slugify(header),
    visual,
    narration,
    bullets,
    terminal,
    cues: deriveCues(sentences, durationMs),
    durationMs,
  };
}

export function compileScript(markdown: string, sourceName = "script"): VideoScript {
  const { fields, body } = parseFrontMatter(markdown);
  const h1 = body.split("\n").map((l) => l.match(H1)?.[1]).find(Boolean);

  const title = fields["title"] ?? h1;
  if (!title) {
    throw new PlanError(`${sourceName}: needs a "# Title" heading or "title:" front matter`);
  }

  const lines = body.split("\n");
  const chapterStarts: { name: string; index: number }[] = [];
  lines.forEach((line, index) => {
    const m = line.match(H2);
    if (m) chapterStarts.push({ name: m[1]!, index });
  });

  if (chapterStarts.length === 0) {
    throw new PlanError(`${sourceName}: no "## Chapter" headings found`);
  }

  const chapters = chapterStarts.map(({ name, index }, i) => {
    const end = i + 1 < chapterStarts.length ? chapterStarts[i + 1]!.index : lines.length;
    return parseChapter(name, lines.slice(index + 1, end));
  });

  // Composition ids must be valid Remotion ids: letters, digits, hyphens.
  const id = fields["id"] ?? slugify(title);
  if (!/^[a-zA-Z0-9-]+$/.test(id)) {
    throw new PlanError(
      `${sourceName}: composition id "${id}" must be letters, digits, or hyphens`
    );
  }

  return {
    id,
    title,
    chapters,
    durationMs: chapters.reduce((sum, c) => sum + c.durationMs, 0),
  };
}

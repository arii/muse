import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  chapterVoiceoverFile,
  deriveCues,
  fitChapterToAudio,
  fitCuesToDuration,
  framesToMs,
  msToFrames,
  splitSentences,
} from "../src/timing";
import type { Chapter, Cue } from "../src/script-types";

function chapterOf(narration: string, durationMs: number): Chapter {
  return {
    name: "Chapter",
    slug: "chapter",
    visual: "title",
    narration,
    bullets: [],
    terminal: [],
    cues: deriveCues(splitSentences(narration), durationMs),
    durationMs,
  };
}

describe("splitSentences", () => {
  it("splits a paragraph into sentences, keeping punctuation", () => {
    assert.deepEqual(splitSentences("One fact here. Two more facts follow! Really?"), [
      "One fact here.",
      "Two more facts follow!",
      "Really?",
    ]);
  });

  it("keeps a single sentence whole", () => {
    assert.deepEqual(splitSentences("Just one."), ["Just one."]);
  });

  it("handles text without terminal punctuation as one unit", () => {
    assert.deepEqual(splitSentences("no punctuation at all"), [
      "no punctuation at all",
    ]);
  });

  it("returns nothing for empty text", () => {
    assert.deepEqual(splitSentences(""), []);
  });

  it("never splits inside a decimal number (regression)", () => {
    const text = "Runs finished in 3.5 seconds. Version 1.2 shipped.";
    assert.deepEqual(splitSentences(text), [
      "Runs finished in 3.5 seconds.",
      "Version 1.2 shipped.",
    ]);
  });

  it("round-trips: sentences joined equal the source text", () => {
    const text = "One here. Two there! And a 2.5 tail?";
    assert.equal(splitSentences(text).join(" "), text);
  });
});

describe("deriveCues (sentence-level)", () => {
  it("gives one cue per sentence unit", () => {
    const cues = deriveCues(["First sentence.", "Second sentence.", "Third."], 6000);
    assert.equal(cues.length, 3);
    assert.deepEqual(
      cues.map((c) => c.text),
      ["First sentence.", "Second sentence.", "Third."]
    );
  });

  it("spends more time on wordier cues", () => {
    const cues = deriveCues(["Go.", "A considerably longer sentence with many more words."], 5000);
    const short = cues[0]!.endMs - cues[0]!.startMs;
    const long = cues[1]!.endMs - cues[1]!.startMs;
    assert.ok(long > short, `long ${long} should exceed short ${short}`);
  });

  it("anchors at zero, chains cues, and pins the end exactly", () => {
    const cues = deriveCues(["one.", "two.", "three."], 4000);
    assert.equal(cues[0]!.startMs, 0);
    for (let i = 1; i < cues.length; i++) {
      assert.equal(cues[i]!.startMs, cues[i - 1]!.endMs);
    }
    assert.equal(cues[cues.length - 1]!.endMs, 4000);
  });

  it("returns no cues for a silent chapter", () => {
    assert.deepEqual(deriveCues([], 3000), []);
  });
});

describe("fitCuesToDuration", () => {
  const cues: Cue[] = [
    { text: "a", startMs: 0, endMs: 1000 },
    { text: "b", startMs: 1000, endMs: 2000 },
  ];

  it("scales cue timings to the measured duration", () => {
    const fitted = fitCuesToDuration(cues, 4000);
    assert.equal(fitted[0]!.startMs, 0);
    assert.equal(fitted[0]!.endMs, 2000);
    assert.equal(fitted[1]!.startMs, 2000);
    assert.equal(fitted[1]!.endMs, 4000);
  });

  it("keeps cue text verbatim while re-timing", () => {
    const fitted = fitCuesToDuration(cues, 3000);
    assert.deepEqual(
      fitted.map((c) => c.text),
      ["a", "b"]
    );
  });

  it("handles a zero-duration input by re-deriving evenly", () => {
    const zero: Cue[] = [
      { text: "a", startMs: 0, endMs: 0 },
      { text: "b", startMs: 0, endMs: 0 },
    ];
    const fitted = fitCuesToDuration(zero, 2000);
    assert.equal(fitted[fitted.length - 1]!.endMs, 2000);
  });

  it("passes an empty cue list through", () => {
    assert.deepEqual(fitCuesToDuration([], 1000), []);
  });
});

describe("fitChapterToAudio", () => {
  it("keeps the plan estimate when no audio was measured (silent card)", () => {
    const chapter = chapterOf("", 3000);
    assert.equal(fitChapterToAudio(chapter, null), chapter);
  });

  it("re-times the chapter to the measured narration duration", () => {
    const chapter = chapterOf("Some narration goes here.", 10000);
    const fitted = fitChapterToAudio(chapter, 1.9); // 57 frames at 30 fps
    assert.equal(fitted.durationMs, framesToMs(57));
    assert.ok(Math.abs(fitted.durationMs - 1900) <= 1000 / 30);
  });

  it("frame-aligns the fitted duration", () => {
    const fitted = fitChapterToAudio(chapterOf("Words to speak.", 8000), 2.777);
    assert.equal(fitted.durationMs, framesToMs(msToFrames(fitted.durationMs)));
  });

  it("re-derives cues from the same narration so text cannot drift", () => {
    const narration = "First sentence here. Second one follows.";
    const fitted = fitChapterToAudio(chapterOf(narration, 9000), 2.0);
    assert.deepEqual(
      fitted.cues.map((c) => c.text),
      ["First sentence here.", "Second one follows."]
    );
    assert.equal(fitted.cues[fitted.cues.length - 1]!.endMs, fitted.durationMs);
    // Re-derived from the same spelled-as-read narration as the TTS input.
    assert.equal(fitted.narration, chapterOf(narration, 9000).narration);
  });

  it("ignores audio for a chapter without narration", () => {
    const chapter = chapterOf("", 3000);
    assert.equal(fitChapterToAudio(chapter, 2.5), chapter);
  });
});

describe("frame helpers", () => {
  it("round-trips frame-aligned durations", () => {
    for (const frames of [1, 30, 57, 300, 3600]) {
      assert.equal(msToFrames(framesToMs(frames)), frames);
    }
  });

  it("never emits fewer than one frame", () => {
    assert.equal(msToFrames(0), 1);
  });
});

describe("chapterVoiceoverFile", () => {
  it("matches the TTS output layout relative to public/", () => {
    assert.equal(
      chapterVoiceoverFile("demo", "two-views-one-script"),
      "voiceover/demo/two-views-one-script.wav"
    );
  });
});

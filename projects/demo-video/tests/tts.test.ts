import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, it } from "node:test";
import {
  applyCacheEnv,
  countWords,
  synthesizeScript,
  synthSecondsPer100Words,
  voiceoverPath,
  type SynthesizedAudio,
  type TtsHandle,
} from "../scripts/tts";
import type { Chapter, VideoScript } from "../src/script-types";

function chapterWith(slug: string, narration: string): Chapter {
  return {
    name: slug,
    slug,
    visual: "title",
    narration,
    bullets: [],
    terminal: [],
    cues: [],
    durationMs: 1000,
  };
}

// Fake RawAudio: duration derived from the sample count at 24 kHz, like the
// real thing; save writes a placeholder so file assertions have a target.
function fakeAudio(seconds: number): SynthesizedAudio {
  return {
    audio: { length: Math.round(24000 * seconds) },
    sampling_rate: 24000,
    save: (p) => fs.writeFileSync(p, "fake wav"),
  };
}

function handleOf(
  generate: (text: string) => Promise<SynthesizedAudio>
): TtsHandle {
  return { generate };
}

function scriptOf(chapters: Chapter[]): VideoScript {
  return {
    id: "tts-test",
    title: "TTS Test",
    chapters,
    durationMs: chapters.reduce((n, c) => n + c.durationMs, 0),
  };
}

function tempOutDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "demo-video-tts-"));
}

describe("synthesizeScript", () => {
  let outDir: string;

  afterEach(() => {
    if (outDir) fs.rmSync(outDir, { recursive: true, force: true });
  });

  it("writes one file per narrated chapter and skips silent chapters", async () => {
    outDir = tempOutDir();
    const script = scriptOf([
      chapterWith("intro", "Hello from the intro."),
      chapterWith("silent-card", ""),
      chapterWith("closing", "And that is the demo."),
    ]);
    const handle = handleOf(async (text) => fakeAudio(text.length / 100));
    const results = await synthesizeScript(script, "af_heart", {
      outDir,
      loadHandle: () => Promise.resolve(handle),
    });

    assert.equal(results.length, 2);
    assert.deepEqual(
      results.map((r) => r.chapter),
      ["intro", "closing"]
    );
    for (const r of results) {
      assert.ok(fs.existsSync(r.file), `expected ${r.file} to exist`);
    }
    assert.ok(
      !fs.existsSync(path.join(outDir, "tts-test", "silent-card.wav")),
      "silent chapters get no audio file"
    );
  });

  it("derives the exact duration from samples / sampling rate", async () => {
    outDir = tempOutDir();
    const script = scriptOf([chapterWith("timed", "Two seconds of audio.")]);
    const handle = handleOf(() => Promise.resolve(fakeAudio(2)));
    const [result] = await synthesizeScript(script, "af_heart", {
      outDir,
      loadHandle: () => Promise.resolve(handle),
    });
    assert.equal(result!.durationSec, 2);
    assert.equal(result!.words, 4);
    assert.ok(result!.synthMs >= 0);
  });

  it("names the chapter when synthesis fails", async () => {
    outDir = tempOutDir();
    const script = scriptOf([
      chapterWith("fine", "This one works."),
      chapterWith("broken", "This one blows up."),
    ]);
    const handle = handleOf(async (text) => {
      if (text.includes("blows up")) {
        throw new Error("model exploded");
      }
      return fakeAudio(1);
    });
    await assert.rejects(
      synthesizeScript(script, "af_heart", {
        outDir,
        loadHandle: () => Promise.resolve(handle),
      }),
      (err: Error) => {
        assert.match(err.message, /chapter "broken" \(broken\)/);
        assert.match(err.message, /model exploded/);
        return true;
      }
    );
  });

  it("writes files under <outDir>/<script-id>/<chapter-slug>.wav", async () => {
    outDir = tempOutDir();
    const chapter = chapterWith("opening", "Narration.");
    const script = scriptOf([chapter]);
    const handle = handleOf(() => Promise.resolve(fakeAudio(1)));
    const [result] = await synthesizeScript(script, "af_heart", {
      outDir,
      loadHandle: () => Promise.resolve(handle),
    });
    assert.equal(
      result!.file,
      voiceoverPath(script.id, chapter, outDir)
    );
    assert.ok(result!.file.endsWith(path.join("tts-test", "opening.wav")));
  });
});

describe("voiceoverPath", () => {
  it("names voiceover files <script-id>/<chapter-slug>.wav under the cache root", () => {
    const chapter = chapterWith("narrated", "Narration.");
    const script = scriptOf([chapter]);
    assert.ok(
      voiceoverPath(script.id, chapter).includes(
        path.join(script.id, "narrated.wav")
      )
    );
  });
});

describe("countWords", () => {
  it("counts whitespace-separated words", () => {
    assert.equal(countWords("one two three"), 3);
    assert.equal(countWords("  padded   text  "), 2);
    assert.equal(countWords(""), 0);
  });
});

describe("synthSecondsPer100Words", () => {
  it("scales total synthesis time to 100 words", () => {
    const per100 = synthSecondsPer100Words([
      { chapter: "a", file: "a", durationSec: 1, words: 100, synthMs: 10_000 },
      { chapter: "b", file: "b", durationSec: 1, words: 300, synthMs: 15_000 },
    ]);
    // 25 s of synthesis across 400 words -> 6.25 s per 100 words.
    assert.equal(per100, 6.25);
  });

  it("returns null when nothing was synthesized", () => {
    assert.equal(synthSecondsPer100Words([]), null);
  });
});

describe("applyCacheEnv", () => {
  it("keeps remote models allowed while the cache is cold", () => {
    const env = { cacheDir: "", allowRemoteModels: true };
    applyCacheEnv(env, "/cache/kokoro", false);
    assert.equal(env.cacheDir, "/cache/kokoro");
    assert.equal(env.allowRemoteModels, true);
  });

  it("disables remote models in offline mode", () => {
    const env = { cacheDir: "", allowRemoteModels: true };
    applyCacheEnv(env, "/cache/kokoro", true);
    assert.equal(env.allowRemoteModels, false);
  });
});

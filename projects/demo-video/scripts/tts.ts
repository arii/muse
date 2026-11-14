// Offline TTS via kokoro-js (Kokoro-82M, Apache-2.0). One audio file per
// chapter into public/voiceover/<comp>/<chapter>.wav, following the in-repo
// voiceover.md scene-per-audio-file pattern with the ElevenLabs fetch
// replaced by local synthesis. No API keys anywhere.

import fs from "node:fs";
import path from "node:path";
import {
  KOKORO_CACHE_DIR,
  KOKORO_DTYPE,
  KOKORO_MODEL_ID,
  KOKORO_PIN,
  TTS_SPEED,
  VOICEOVER_DIR,
} from "./config";
import { kokoroModelPresent } from "./preflight";
import type { Chapter, VideoScript } from "../src/script-types";

// Structural shape of kokoro-js RawAudio that synthesis needs: the sample
// buffer (for the exact duration), the sampling rate, and a wav writer.
export type SynthesizedAudio = {
  audio: { length: number };
  sampling_rate: number;
  save: (path: string) => void;
};

export type SynthesisResult = {
  chapter: string;
  file: string;
  // Exact duration from the synthesized sample count (samples / rate) — not
  // an estimate, so per-chapter scene timing can be checked against ffprobe.
  durationSec: number;
  words: number;
  synthMs: number;
};

export type TtsHandle = {
  generate: (text: string) => Promise<SynthesizedAudio>;
};

export type ModelCacheEnv = {
  cacheDir: string;
  allowRemoteModels: boolean;
};

// Redirect the transformers.js cache into .cache/kokoro. Pure so tests can
// assert the offline behavior; configureModelCache applies it to the real env.
export function applyCacheEnv(
  env: ModelCacheEnv,
  cacheDir: string,
  offline: boolean
): void {
  env.cacheDir = cacheDir;
  // Offline mode: with the cache populated, transformers.js serves every
  // file from disk and never constructs a remote URL.
  env.allowRemoteModels = !offline;
}

// Must run before the model loads.
export async function configureModelCache(offline = false): Promise<void> {
  const { env } = await import("@huggingface/transformers");
  applyCacheEnv(env as unknown as ModelCacheEnv, KOKORO_CACHE_DIR, offline);
}

// With the model cache populated, synthesis runs fully offline: transformers.js
// is pinned to local files. A cold cache fetches once (`npm run setup` does
// this), then every later run is offline.
export function cacheIsPopulated(cacheDir: string): boolean {
  if (!fs.existsSync(cacheDir)) return false;
  const files = fs.readdirSync(cacheDir, { recursive: true }).map(String);
  return kokoroModelPresent(files, KOKORO_MODEL_ID.split("/").pop()!);
}

let kokoroModule: typeof import("kokoro-js") | null = null;

// --- model cache pinning ---------------------------------------------------
// The cache is pinned by version: config's KOKORO_PIN is written to
// pin.json after a successful fetch and verified before every load, so
// synthesis never silently mixes model versions.

export type CachePin = {
  modelId: string;
  dtype: string;
  kokoroJs: string;
  transformers: string;
};

export function pinsMatch(cached: CachePin, expected: CachePin): boolean {
  return (
    cached.modelId === expected.modelId &&
    cached.dtype === expected.dtype &&
    cached.kokoroJs === expected.kokoroJs &&
    cached.transformers === expected.transformers
  );
}

function pinPath(cacheDir: string): string {
  return path.join(cacheDir, "pin.json");
}

export function readCachePin(cacheDir: string): CachePin | null {
  try {
    return JSON.parse(fs.readFileSync(pinPath(cacheDir), "utf8")) as CachePin;
  } catch {
    return null; // no pin yet — first fetch writes it
  }
}

function assertCachePin(cacheDir: string): void {
  const cached = readCachePin(cacheDir);
  if (cached && !pinsMatch(cached, KOKORO_PIN)) {
    throw new Error(
      `kokoro model cache at ${cacheDir} was fetched for ${cached.modelId}/${cached.dtype} ` +
        `(kokoro-js ${cached.kokoroJs}) but config pins ${KOKORO_PIN.modelId}/${KOKORO_PIN.dtype} ` +
        `(kokoro-js ${KOKORO_PIN.kokoroJs}). Delete the directory and re-run npm run setup.`
    );
  }
}

function writeCachePin(cacheDir: string): void {
  fs.mkdirSync(cacheDir, { recursive: true });
  fs.writeFileSync(
    pinPath(cacheDir),
    JSON.stringify(KOKORO_PIN, null, 2) + "\n"
  );
}

export async function loadTts(voice: string): Promise<TtsHandle> {
  const offline = cacheIsPopulated(KOKORO_CACHE_DIR);
  if (!kokoroModule) {
    await configureModelCache(offline);
    kokoroModule = await import("kokoro-js");
  }
  assertCachePin(KOKORO_CACHE_DIR);
  const { KokoroTTS } = kokoroModule;
  const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
    dtype: KOKORO_DTYPE,
    device: "cpu",
  });
  writeCachePin(KOKORO_CACHE_DIR);
  return {
    generate: (text: string) =>
      tts.generate(text, {
        voice: validateVoice(tts, voice),
        speed: TTS_SPEED,
      }),
  };
}

// kokoro-js types the voice as a literal union of the bundled voice ids;
// validate the string at runtime against the loaded model's voice list so a
// bad --voice fails with the known options, not a synthesis error.
function validateVoice(
  tts: import("kokoro-js").KokoroTTS,
  voice: string
): keyof typeof tts.voices {
  if (!(voice in tts.voices)) {
    throw new Error(
      `unknown kokoro voice "${voice}" (known: ${Object.keys(tts.voices).join(", ")})`
    );
  }
  return voice as keyof typeof tts.voices;
}

export function voiceoverPath(
  scriptId: string,
  chapter: Chapter,
  outDir: string = VOICEOVER_DIR
): string {
  return path.join(outDir, scriptId, `${chapter.slug}.wav`);
}

export function countWords(text: string): number {
  const words = text.trim().match(/\S+/g);
  return words ? words.length : 0;
}

// Synthesis throughput in seconds per 100 words — the Piper fallback decision
// metric. Returns null when nothing was synthesized.
export function synthSecondsPer100Words(
  results: SynthesisResult[]
): number | null {
  const words = results.reduce((n, r) => n + r.words, 0);
  if (words === 0) return null;
  const seconds = results.reduce((n, r) => n + r.synthMs, 0) / 1000;
  return (seconds / words) * 100;
}

export type SynthesizeOptions = {
  // Inject a handle in tests; defaults to loading the real kokoro model.
  loadHandle?: () => Promise<TtsHandle>;
  // Defaults to the project voiceover dir.
  outDir?: string;
};

// Synthesize every chapter that has narration; chapters without narration are
// silent cards and get no audio file. Synthesis failing for one chapter fails
// the run with the chapter named — never a silent gap.
export async function synthesizeScript(
  script: VideoScript,
  voice: string,
  options: SynthesizeOptions = {}
): Promise<SynthesisResult[]> {
  const outDir = options.outDir ?? VOICEOVER_DIR;
  const load = options.loadHandle ?? (() => loadTts(voice));
  const tts = await load();
  const results: SynthesisResult[] = [];
  for (const chapter of script.chapters) {
    if (!chapter.narration) {
      continue;
    }
    const out = voiceoverPath(script.id, chapter, outDir);
    const startedAt = Date.now();
    try {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      const audio = await tts.generate(chapter.narration);
      audio.save(out);
      results.push({
        chapter: chapter.slug,
        file: out,
        durationSec: audio.audio.length / audio.sampling_rate,
        words: countWords(chapter.narration),
        synthMs: Date.now() - startedAt,
      });
    } catch (err) {
      throw new Error(
        `TTS synthesis failed for chapter "${chapter.name}" (${chapter.slug}): ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    }
  }
  return results;
}

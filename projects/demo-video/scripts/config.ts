// Pinned toolkit configuration. Everything versioned lives here so upgrades
// and voice changes are one-line diffs.

import path from "node:path";

export const PROJECT_ROOT = path.resolve(import.meta.dirname, "..");

// kokoro-js model: Kokoro-82M, Apache-2.0 (code and weights), offline.
export const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
export const KOKORO_DTYPE = "q8" as const;

// Version pin for the offline model cache: recorded into .cache/kokoro/pin.json
// after a successful fetch and verified on every load. Changing any value
// invalidates the cache — synthesis refuses to mix model versions; delete
// .cache/kokoro and re-run `npm run setup` to re-fetch. transformers is pinned
// here because tts.ts imports its env directly (it is a real dependency of
// kokoro-js 1.2.1).
export const KOKORO_PIN = {
  modelId: KOKORO_MODEL_ID,
  dtype: KOKORO_DTYPE,
  kokoroJs: "1.2.1",
  transformers: "3.8.1",
} as const;
// Pre-selected, quality-audited English voice — good defaults with zero tuning.
export const KOKORO_VOICE = "af_heart";
export const TTS_SPEED = 1.0;

// Local caches (both gitignored under .cache/).
export const KOKORO_CACHE_DIR = path.join(PROJECT_ROOT, ".cache", "kokoro");
export const BROWSER_CACHE_DIR = path.join(PROJECT_ROOT, ".cache", "browser");

// Where compiled script plans land for src/Root.tsx to discover.
export const GENERATED_DIR = path.join(PROJECT_ROOT, "src", "generated");

// Where per-chapter narration audio is written for staticFile().
export const VOICEOVER_DIR = path.join(PROJECT_ROOT, "public", "voiceover");

// Silent-estimate pacing: ~160 spoken words per minute plus a settle pad.
export const WORDS_PER_SECOND = 160 / 60;
export const CHAPTER_MIN_MS = 3000;
export const CHAPTER_PAD_MS = 600;

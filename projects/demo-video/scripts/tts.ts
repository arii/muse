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
  TTS_SPEED,
  VOICEOVER_DIR,
} from "./config";
import type { Chapter, VideoScript } from "../src/script-types";

export type SynthesisResult = {
  chapter: string;
  file: string;
  durationSec: number | null;
};

let kokoroModule: typeof import("kokoro-js") | null = null;

export type TtsHandle = {
  generate: (text: string) => Promise<{ save: (p: string) => void }>;
};

// Redirect the transformers.js cache into .cache/kokoro (it defaults to a
// path inside node_modules). Must run before the model loads; after that,
// synthesis runs fully offline.
export async function configureModelCache(): Promise<void> {
  const { env } = await import("@huggingface/transformers");
  env.cacheDir = KOKORO_CACHE_DIR;
  env.allowRemoteModels = true;
}

// The model download happens on first load and is cached under .cache/kokoro;
// afterwards synthesis runs fully offline.
export async function loadTts(voice: string): Promise<TtsHandle> {
  if (!kokoroModule) {
    await configureModelCache();
    kokoroModule = await import("kokoro-js");
  }
  const { KokoroTTS } = kokoroModule;
  const tts = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
    dtype: KOKORO_DTYPE,
    device: "cpu",
  });
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

export function voiceoverPath(scriptId: string, chapter: Chapter): string {
  return path.join(VOICEOVER_DIR, scriptId, `${chapter.slug}.wav`);
}

export function voiceoverExists(scriptId: string, chapter: Chapter): boolean {
  return fs.existsSync(voiceoverPath(scriptId, chapter));
}

// Synthesize every chapter that has narration; chapters without narration are
// silent cards and get no audio file. Synthesis failing for one chapter fails
// the run with the chapter named — never a silent gap.
export async function synthesizeScript(
  script: VideoScript,
  voice: string
): Promise<SynthesisResult[]> {
  const tts = await loadTts(voice);
  const results: SynthesisResult[] = [];
  for (const chapter of script.chapters) {
    if (!chapter.narration) {
      continue;
    }
    const out = voiceoverPath(script.id, chapter);
    try {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      const audio = await tts.generate(chapter.narration);
      audio.save(out);
      results.push({
        chapter: chapter.slug,
        file: out,
        durationSec: null,
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

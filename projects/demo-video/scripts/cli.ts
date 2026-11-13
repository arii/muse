// demo-video CLI: plan → tts → render, one narration script in, one video out.
// Long flags and a usage block, matching repo CLI conventions.

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { compileScript } from "./plan";
import { synthesizeScript } from "./tts";
import {
  GENERATED_DIR,
  KOKORO_VOICE,
  PROJECT_ROOT,
} from "./config";
import { BROWSER_CACHE_ROOTS, findBrowserExecutable } from "./browser";

export const USAGE = `demo-video — repo + narration script in, demo video out

Usage (npm script aliases, run from projects/demo-video/):
  npm run plan -- <script.md>                compile a script, write src/generated/<id>.json
  npm run tts -- <script.md> [--voice <id>]  synthesize narration only
  npm run render -- <script.md> [flags]      plan -> tts -> render (or --silent)

Options:
  --script <path>            narration script (markdown); alternative to positional
  --voice <id>               kokoro voice (default: ${KOKORO_VOICE})
  --out <path>               output mp4 (default: out/<composition-id>.mp4)
  --concurrency <n>          render concurrency (default: 4)
  --browser-executable <p>   chrome-headless-shell path (default: pre-fetched cache)
  --silent                   skip TTS; render timed silent chapters
  --dry-run                  print what would run, execute nothing
  --help                     this help`;

export type ParsedArgs = {
  command: string | null;
  positional: string[];
  flags: Record<string, string | boolean>;
  error?: string;
  help?: boolean;
};

const VALUE_FLAGS = new Set([
  "script",
  "voice",
  "out",
  "concurrency",
  "browser-executable",
]);

export function parseArgs(argv: string[]): ParsedArgs {
  const parsed: ParsedArgs = { command: null, positional: [], flags: {} };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
      continue;
    }
    if (arg.startsWith("--")) {
      const name = arg.slice(2);
      if (!VALUE_FLAGS.has(name)) {
        // Boolean flags must be known too; anything unknown is an error.
        if (!["silent", "dry-run"].includes(name)) {
          parsed.error = `unknown flag: ${arg}`;
          return parsed;
        }
        parsed.flags[name] = true;
        continue;
      }
      const value = argv[i + 1];
      if (value === undefined || value.startsWith("--")) {
        parsed.error = `flag --${name} requires a value`;
        return parsed;
      }
      parsed.flags[name] = value;
      i++;
      continue;
    }
    if (parsed.command === null) {
      parsed.command = arg;
    } else {
      parsed.positional.push(arg);
    }
  }
  const knownCommands = ["plan", "tts", "render"];
  if (parsed.command !== null && !knownCommands.includes(parsed.command)) {
    parsed.error = `unknown command: ${parsed.command}`;
  }
  return parsed;
}

function fail(message: string): never {
  console.error(`error: ${message}\n\n${USAGE}`);
  process.exit(1);
}

function loadScript(args: ParsedArgs): ReturnType<typeof compileScript> {
  const scriptPath = (args.flags["script"] as string) ?? args.positional[0];
  if (!scriptPath) {
    fail("a narration script is required (positional path or --script <path>)");
  }
  const resolved = path.resolve(PROJECT_ROOT, scriptPath);
  if (!fs.existsSync(resolved)) {
    fail(`script not found: ${resolved}`);
  }
  return compileScript(fs.readFileSync(resolved, "utf8"), scriptPath);
}

export function resolveBrowser(flagValue?: string): string | null {
  if (flagValue) return flagValue;
  return findBrowserExecutable(BROWSER_CACHE_ROOTS(PROJECT_ROOT));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(USAGE);
    return;
  }
  if (args.error) fail(args.error);
  if (!args.command) fail("a command is required (plan | tts | render)");

  if (args.command === "plan") {
    const script = loadScript(args);
    fs.mkdirSync(GENERATED_DIR, { recursive: true });
    const outPath = path.join(GENERATED_DIR, `${script.id}.json`);
    fs.writeFileSync(outPath, JSON.stringify(script, null, 2) + "\n");
    console.log(JSON.stringify(script, null, 2));
    console.error(`plan: wrote ${outPath} (${script.chapters.length} chapters, ${(script.durationMs / 1000).toFixed(1)}s)`);
    return;
  }

  const script = loadScript(args);
  const silent = args.flags["silent"] === true;
  const voice = (args.flags["voice"] as string) ?? KOKORO_VOICE;

  if (args.command === "tts") {
    if (silent) fail("--silent is a render flag; tts always synthesizes");
    const results = await synthesizeScript(script, voice);
    console.error(`tts: ${results.length} chapter file(s) under public/voiceover/${script.id}/`);
    return;
  }

  // render
  if (args.flags["dry-run"] === true) {
    console.log(
      JSON.stringify(
        {
          dryRun: true,
          composition: script.id,
          chapters: script.chapters.map((c) => ({
            slug: c.slug,
            visual: c.visual,
            narration: c.narration,
            durationMs: c.durationMs,
          })),
          wouldRun: silent ? ["plan", "render"] : ["plan", "tts", "render"],
        },
        null,
        2
      )
    );
    return;
  }

  if (!silent) {
    console.error("tts: synthesizing narration...");
    await synthesizeScript(script, voice);
  }

  const out = (args.flags["out"] as string) ?? path.join("out", `${script.id}.mp4`);
  const browser = resolveBrowser(args.flags["browser-executable"] as string | undefined);
  const concurrency = Number(args.flags["concurrency"] ?? 4);

  const renderArgs = [
    "remotion",
    "render",
    script.id,
    path.resolve(PROJECT_ROOT, out),
    "--concurrency",
    String(concurrency),
  ];
  if (browser) {
    renderArgs.push("--browser-executable", browser);
  }
  console.error(`render: ${renderArgs.join(" ")}`);
  execFileSync("npx", renderArgs, {
    cwd: PROJECT_ROOT,
    stdio: "inherit",
  });
}

// Run only when executed directly — importing this module (tests) must not
// trigger the CLI.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err: unknown) => {
    console.error(
      `demo-video: ${err instanceof Error ? err.message : String(err)}`
    );
    process.exit(1);
  });
}

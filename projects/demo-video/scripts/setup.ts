// npm run setup — install-time environment pre-flight and browser pre-fetch.
//
// 1. Verifies node, ffmpeg, the pre-fetched chrome-headless-shell, and the
//    cached kokoro model.
// 2. Pre-fetches the browser: `remotion browser ensure` first; on the
//    documented downloader timeout, extracts the Chrome for Testing URL or
//    version Remotion prints and downloads it manually into .cache/browser/.
// 3. Pre-fetches the kokoro model into .cache/kokoro so later synthesis is
//    fully offline.
// Prints a PASS/FAIL report and exits non-zero if anything fails.

import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import {
  BROWSER_CACHE_DIR,
  KOKORO_CACHE_DIR,
  KOKORO_DTYPE,
  KOKORO_MODEL_ID,
  PROJECT_ROOT,
} from "./config";
import {
  BROWSER_MISSING_INSTRUCTIONS,
  checkFfmpeg,
  checkNode,
  formatPreflightReport,
  kokoroModelPresent,
  parseBrowserDownloadHint,
  type Check,
} from "./preflight";
import { configureModelCache } from "./tts";
import { findBrowserExecutable } from "./browser";

const ENGINES = JSON.parse(
  fs.readFileSync(path.join(PROJECT_ROOT, "package.json"), "utf8")
)["engines"]["node"] as string;

function run(cwd: string, cmd: string, args: string[]): {
  ok: boolean;
  output: string;
} {
  try {
    const output = execFileSync(cmd, args, {
      cwd,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { ok: true, output };
  } catch (err) {
    const e = err as { stdout?: string; stderr?: string; message?: string };
    return {
      ok: false,
      output: [e.stdout, e.stderr, e.message].filter(Boolean).join("\n"),
    };
  }
}

// Download chrome-headless-shell manually into .cache/browser/. The hint may
// be a full URL (Remotion prints it) or just a version — the CDN layout is
// predictable from a version.
async function manualBrowserFetch(
  hint: { url?: string; version?: string }
): Promise<string | null> {
  const url =
    hint.url ??
    (hint.version
      ? `https://storage.googleapis.com/chrome-for-testing-public/${hint.version}/linux64/chrome-headless-shell-linux64.zip`
      : undefined);
  if (!url) return null;
  fs.mkdirSync(BROWSER_CACHE_DIR, { recursive: true });
  const zipPath = path.join(BROWSER_CACHE_DIR, "chrome-headless-shell.zip");
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    return null;
  }
  fs.writeFileSync(zipPath, Buffer.from(await res.arrayBuffer()));
  const unzip = run(BROWSER_CACHE_DIR, "unzip", ["-o", zipPath]);
  if (!unzip.ok) return null;
  const binary = path.join(
    BROWSER_CACHE_DIR,
    "chrome-headless-shell-linux64",
    "chrome-headless-shell"
  );
  if (!fs.existsSync(binary)) return null;
  fs.chmodSync(binary, 0o755);
  fs.rmSync(zipPath, { force: true });
  return binary;
}

// First load downloads and caches the model; later runs are offline.
async function fetchKokoroModel(): Promise<boolean> {
  try {
    await configureModelCache();
    const { KokoroTTS } = await import("kokoro-js");
    await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
      dtype: KOKORO_DTYPE,
      device: "cpu",
    });
    return true;
  } catch (err) {
    console.error(
      `kokoro model fetch failed: ${err instanceof Error ? err.message : String(err)}`
    );
    return false;
  }
}

async function main(): Promise<void> {
  const checks: Check[] = [];

  // node
  const nodeCheck = checkNode(process.version, ENGINES);
  checks.push(nodeCheck);

  // ffmpeg
  const ffmpeg = run(PROJECT_ROOT, "ffmpeg", ["-version"]);
  checks.push(checkFfmpeg(ffmpeg.output));

  // browser — try remotion's ensure, fall back to a manual CDN fetch.
  let browserCheck: Check;
  const ensure = run(PROJECT_ROOT, "npx", ["remotion", "browser", "ensure"]);
  if (ensure.ok) {
    browserCheck = {
      name: "browser",
      pass: true,
      detail: "remotion browser ensure",
    };
  } else {
    fs.mkdirSync(BROWSER_CACHE_DIR, { recursive: true });
    fs.writeFileSync(path.join(BROWSER_CACHE_DIR, "setup.log"), ensure.output);
    const hint = parseBrowserDownloadHint(ensure.output);
    const binary = await manualBrowserFetch(hint);
    browserCheck = binary
      ? { name: "browser", pass: true, detail: binary }
      : {
          name: "browser",
          pass: false,
          detail: `${BROWSER_MISSING_INSTRUCTIONS}\n${ensure.output}`,
        };
  }
  if (browserCheck.pass) {
    const resolved = findBrowserExecutable([
      BROWSER_CACHE_DIR,
      path.join(PROJECT_ROOT, "node_modules", ".remotion"),
    ]);
    if (resolved) {
      browserCheck.detail = resolved;
    }
  }
  checks.push(browserCheck);

  // kokoro model — already cached? Otherwise fetch once.
  const cacheFiles = fs.existsSync(KOKORO_CACHE_DIR)
    ? fs.readdirSync(KOKORO_CACHE_DIR, { recursive: true }).map(String)
    : [];
  const modelName = KOKORO_MODEL_ID.split("/").pop()!;
  let kokoroCheck: Check;
  if (kokoroModelPresent(cacheFiles, modelName)) {
    kokoroCheck = { name: "kokoro model", pass: true, detail: KOKORO_CACHE_DIR };
  } else {
    const ok = await fetchKokoroModel();
    kokoroCheck = ok
      ? { name: "kokoro model", pass: true, detail: `fetched ${KOKORO_MODEL_ID}` }
      : {
          name: "kokoro model",
          pass: false,
          detail: `could not fetch ${KOKORO_MODEL_ID} into ${KOKORO_CACHE_DIR}`,
        };
  }
  checks.push(kokoroCheck);

  console.log(formatPreflightReport(checks));
  if (!checks.every((c) => c.pass)) {
    process.exit(1);
  }
}

main().catch((err: unknown) => {
  console.error(`setup: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});

/// <reference lib="es2022.error" />
import fs from "node:fs";
import path from "node:path";
import { spawn, execFileSync } from "node:child_process";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import { chromium, type Page } from "playwright-core";
import { findBrowserExecutable, BROWSER_CACHE_ROOTS } from "./browser";
import { PROJECT_ROOT } from "./config";
import { serveDirectory } from "./capture-server";

declare global {
  interface Window {
    __cinema?: { ready: boolean; duration: number; render(t: number): void | Promise<void> };
  }
}

export const CAPTURE_USAGE = `Frame-stepped Cinema capture
  npm run capture -- --url <url> --out <file.mp4> [options]
  npm run capture -- --serve <directory> --path '/?cinema=1&capture=1' --out <file.mp4>
Options: --duration <seconds> (default: __cinema.duration), --fps <n> (30),
  --size <width>x<height> (1280x720), --stills <5,18>,
  --timeout-ms <n> (30000), --browser-executable <path>
Stills are saved beside the MP4 as <name>_<seconds>s.png.`;

export function captureOptions(args: string[]) {
  const { values } = parseArgs({ args, options: {
    url: { type: "string" }, serve: { type: "string" }, path: { type: "string" },
    out: { type: "string" }, duration: { type: "string" }, fps: { type: "string", default: "30" },
    size: { type: "string", default: "1280x720" }, stills: { type: "string" },
    "timeout-ms": { type: "string", default: "30000" }, "browser-executable": { type: "string" },
    help: { type: "boolean" },
  } });
  if (values.help) return null;
  if (Boolean(values.url) === Boolean(values.serve)) throw new Error("Provide exactly one of --url or --serve");
  if (!values.out?.endsWith(".mp4")) throw new Error("--out must name an .mp4 file");
  const size = values.size!.match(/^(\d+)x(\d+)$/);
  if (!size || size.slice(1).some(n => Number(n) < 2 || Number(n) > 8192 || Number(n) % 2)) {
    throw new Error("--size requires even dimensions between 2 and 8192, e.g. 1280x720");
  }
  const fps = Number(values.fps), timeout = Number(values["timeout-ms"]);
  const duration = values.duration === undefined ? undefined : Number(values.duration);
  if (!Number.isInteger(fps) || fps < 1 || fps > 120) throw new Error("--fps must be an integer between 1 and 120");
  if (!Number.isFinite(timeout) || timeout <= 0) throw new Error("--timeout-ms must be positive");
  if (duration !== undefined && (!Number.isFinite(duration) || duration <= 0)) throw new Error("--duration must be positive");
  const stills = values.stills === undefined ? [] : values.stills.split(",").map(Number);
  if (stills.some(n => !Number.isFinite(n) || n < 0)) throw new Error("--stills requires nonnegative seconds");
  if (values.path && (!values.path.startsWith("/") || values.path.startsWith("//"))) throw new Error("--path must be a local absolute URL path");
  if (values.url && !["http:", "https:"].includes(new URL(values.url).protocol)) throw new Error("--url must use HTTP or HTTPS");
  return { ...values, out: path.resolve(values.out), fps, duration, timeout, stills,
    width: Number(size[1]), height: Number(size[2]) };
}

export async function captureMain(args: string[]): Promise<void> {
  const parsed = captureOptions(args);
  if (!parsed) { console.log(CAPTURE_USAGE); return; }
  const options = parsed;
  const executablePath = captureBrowser(options["browser-executable"]);
  if (!executablePath) throw new Error("Browser not found; run npm run setup or pass --browser-executable");
  execFileSync("ffmpeg", ["-version"], { stdio: "ignore" });
  const hosted = options.serve ? await serveDirectory(options.serve) : null;
  let browser;
  let encoder: ReturnType<typeof spawn> | undefined;
  try {
    const url = options.path ? new URL(options.path, hosted?.url ?? options.url).href : options.url ?? hosted!.url;
    browser = await chromium.launch({ executablePath, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars", "--force-device-scale-factor=1", "--disable-gpu-vsync"] });
    const page = await browser.newPage({ viewport: { width: options.width, height: options.height }, deviceScaleFactor: 1 });
    page.setDefaultTimeout(options.timeout);
    page.on("pageerror", error => console.error(`page error: ${error.message}`));
    await page.goto(url, { timeout: options.timeout, waitUntil: "load" });
    await waitForCinema(page, options.timeout, url);
    await page.evaluate(() => document.fonts.ready);
    const duration = options.duration ?? await page.evaluate(() => window.__cinema!.duration);
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("__cinema.duration must be positive (or provide --duration)");
    if (options.stills.some(n => n >= duration)) throw new Error("--stills times must be inside the capture duration");
    fs.mkdirSync(path.dirname(options.out), { recursive: true });
    encoder = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(options.fps), "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", options.out], { stdio: ["pipe", "inherit", "inherit"] });
    const encoded = new Promise<void>((resolve, reject) => {
      encoder!.once("error", reject);
      encoder!.once("close", code => code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`)));
    });
    const total = Math.ceil(duration * options.fps), started = Date.now();
    async function* frames() {
      for (let i = 0; i < total; i++) {
        await page.evaluate(t => window.__cinema!.render(t), i / options.fps);
        yield await page.screenshot({ type: "jpeg", quality: 94, animations: "disabled" });
        if (i % options.fps === 0) console.log(`capture: frame ${i}/${total} (${((Date.now() - started) / 1000).toFixed(1)}s elapsed)`);
      }
    }
    await Promise.all([pipeline(Readable.from(frames(), { objectMode: false }), encoder.stdin!), encoded]);
    for (const second of options.stills) {
      const file = options.out.replace(/\.mp4$/, `_${second}s.png`);
      await page.evaluate(t => window.__cinema!.render(t), second);
      await page.screenshot({ path: file, type: "png", animations: "disabled" });
      console.log(`still: ${file}`);
    }
    console.log(`capture: ${options.out} (${total} frames, ${options.fps} fps, ${(total / options.fps).toFixed(3)}s)`);
  } finally {
    encoder?.kill();
    await browser?.close();
    if (hosted) await new Promise<void>((resolve, reject) => hosted.server.close(error => error ? reject(error) : resolve()));
  }
}

export function captureBrowser(explicit?: string) {
  return explicit ?? process.env.CHROME_PATH ?? findBrowserExecutable(BROWSER_CACHE_ROOTS(PROJECT_ROOT));
}

export async function waitForCinema(page: Pick<Page, "waitForFunction">, timeout: number, url: string) {
  try {
    await page.waitForFunction(() => window.__cinema?.ready === true && typeof window.__cinema.render === "function", null, { timeout });
  } catch (cause) {
    const message = cause instanceof Error ? cause.message : String(cause);
    throw new Error(`Page never exposed ready window.__cinema.render(t) within ${timeout}ms: ${url}: ${message}`, { cause });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  captureMain(process.argv.slice(2)).catch((error: unknown) => {
    console.error(`capture: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
}

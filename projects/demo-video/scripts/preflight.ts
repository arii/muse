// Preflight checks — pure logic, unit-tested; setup.ts wires them to live
// probes. A failing browser check must produce the manual-download
// instruction, not Remotion's raw downloader timeout.

export type Check = {
  name: "node" | "browser" | "kokoro model" | "ffmpeg";
  pass: boolean;
  detail: string;
};

export type Engines = {
  node: string;
};

// Node engines range from the seed project's package.json
// (^20.19.0 || ^22.13.0 || >=24), which the render sandbox must satisfy.
export function nodeVersionSatisfies(
  version: string,
  engines: string
): boolean {
  const coerced = version.replace(/^v/, "");
  const [majorS, minorS, patchS] = coerced.split(".");
  const major = Number(majorS);
  const minor = Number(minorS);
  const patch = Number(patchS);
  for (const clause of engines.split("||").map((c) => c.trim())) {
    const m = clause.match(/^\^(\d+)\.(\d+)\.(\d+)$/);
    if (m) {
      const [, eMajor, eMinor] = m.map(Number) as unknown as number[];
      if (
        major === eMajor &&
        (minor > eMinor || (minor === eMinor && patch >= Number(m[3])))
      ) {
        return true;
      }
      continue;
    }
    const g = clause.match(/^>=(\d+)$/);
    if (g && major >= Number(g[1])) {
      return true;
    }
  }
  return false;
}

export function checkNode(
  version: string,
  engines: string
): Omit<Check, "detail"> & { detail: string } {
  const pass = nodeVersionSatisfies(version, engines);
  return {
    name: "node",
    pass,
    detail: pass ? version : `${version} does not satisfy engines "${engines}"`,
  };
}

// ffmpeg prints its version on the first line: "ffmpeg version 7.1.5-...".
export function checkFfmpeg(versionOutput: string): Check {
  const first = versionOutput.split("\n")[0]?.trim() ?? "";
  const version = first.match(/version\s+(\S+)/)?.[1];
  return {
    name: "ffmpeg",
    pass: Boolean(version),
    detail: version ? `ffmpeg ${version}` : `unrecognized output: ${first || "(empty)"}`,
  };
}

// Remotion prints the exact Chrome for Testing download URL (or build
// version) when its downloader fails. Extract either so setup can fetch it
// manually — the documented workaround for the downloader timeout.
export function parseBrowserDownloadHint(
  output: string
): { url?: string; version?: string } {
  const url = output.match(
    /https:\/\/\S*chrome-for-testing\S*?\/chrome-headless-shell[^\s"']*\.zip/
  )?.[0];
  if (url) return { url };
  const version = output.match(
    /chrome-headless-shell[^\d\n]*?(\d+\.\d+\.\d+\.\d+)/
  )?.[1];
  return version ? { version } : {};
}

// The browser is considered present if any candidate path exists and points
// at a non-empty executable file.
export function resolveBrowserExecutable(
  exists: (p: string) => boolean,
  size: (p: string) => number,
  candidates: string[]
): string | null {
  for (const candidate of candidates) {
    if (exists(candidate) && size(candidate) > 0) {
      return candidate;
    }
  }
  return null;
}

// The kokoro model is present when the cache dir contains at least one ONNX
// weights file.
export function kokoroModelPresent(
  files: string[],
  modelDirName: string
): boolean {
  return files.some(
    (f) =>
      f.includes(modelDirName) || (f.endsWith(".onnx") && f.includes("kokoro"))
  );
}

export function formatPreflightReport(checks: Check[]): string {
  return checks
    .map((c) => `${c.pass ? "PASS" : "FAIL"} ${c.name}: ${c.detail}`)
    .join("\n");
}

export const BROWSER_MISSING_INSTRUCTIONS = [
  "Browser pre-fetch failed. Manual download:",
  "  1. Read the version/URL Remotion prints on failure (see .cache/setup.log).",
  "  2. Download chrome-headless-shell for that build from the Chrome for Testing CDN.",
  "  3. Unzip into projects/demo-video/.cache/browser/ and chmod +x the binary.",
  "  4. Re-run npm run setup, or pass --browser-executable=<path> to render.",
].join("\n");

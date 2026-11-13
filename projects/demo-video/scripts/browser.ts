// Locates the chrome-headless-shell binary under cache roots. Two layouts in
// the wild: Remotion's own node_modules/.remotion/chrome-headless-shell/...
// (after `remotion browser ensure`) and our manual .cache/browser/ layout.

import fs from "node:fs";
import path from "node:path";

export function findBrowserExecutable(roots: string[]): string | null {
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    for (const entry of fs.readdirSync(root, { recursive: true })) {
      const candidate = path.join(root, String(entry));
      try {
        if (
          fs.statSync(candidate).isFile() &&
          path.basename(candidate) === "chrome-headless-shell"
        ) {
          return candidate;
        }
      } catch {
        // stat can race with the OS; keep scanning
      }
    }
  }
  return null;
}

export const BROWSER_CACHE_ROOTS = (projectRoot: string): string[] => [
  path.join(projectRoot, ".cache", "browser"),
  path.join(projectRoot, "node_modules", ".remotion"),
];

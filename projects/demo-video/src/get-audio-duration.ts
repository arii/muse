// Audio duration measurement for calculateMetadata, following the in-repo
// voiceover.md pattern (bundled skill, Remotion 4.0.540): calculateMetadata is
// evaluated inside the composition runtime, where staticFile() resolves to a
// URL served by Remotion (the public/ dir), read via mediabunny's UrlSource.
//
// Returns null when the audio file does not exist yet (silent chapter, or
// narration not synthesized): the caller falls back to the plan estimate.
// A file that exists but cannot be parsed throws — a corrupt file is a real
// failure, not a missing one.

import { Input, ALL_FORMATS, UrlSource } from "mediabunny";
import { staticFile } from "remotion";

// A missing file surfaces as a 404 from Remotion's static server (or ENOENT
// when a source resolves straight to disk); anything else is a real
// parse/measurement failure.
function isMissingFileError(err: unknown): boolean {
  const code = (err as { code?: string } | null)?.code;
  if (code === "ENOENT") return true;
  return /ENOENT|no such file|404|not found/i.test(
    err instanceof Error ? err.message : String(err)
  );
}

export async function getAudioDuration(
  relativeToPublic: string
): Promise<number | null> {
  const input = new Input({
    formats: ALL_FORMATS,
    source: new UrlSource(staticFile(relativeToPublic)),
  });
  try {
    return await input.computeDuration();
  } catch (err) {
    if (isMissingFileError(err)) {
      return null; // not synthesized yet — plan estimate stands in
    }
    throw new Error(
      `narration audio ${relativeToPublic} exists but could not be measured: ${
        err instanceof Error ? err.message : String(err)
      }`
    );
  } finally {
    input.dispose(); // closes the file handle
  }
}

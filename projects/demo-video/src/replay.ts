export type ReplaySegment = { start: number; end: number; speed: number };
export type ReplaySpec = { file: string; segments: ReplaySegment[] };

export function parseReplay(lines: string[]): ReplaySpec {
  const content = lines.map(line => line.trim()).filter(Boolean);
  const file = content.shift();
  if (!file || file.startsWith("/") || /[\\?#:]/.test(file) || file.split("/").some(part => part === ".." || part === "." || !part)) {
    throw new Error("Replay needs a relative clip filename inside public/replay/<id>/");
  }
  const segments = content.map(line => {
    const match = line.match(/^(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?:\s+x(\d+(?:\.\d+)?))?$/);
    if (!match) throw new Error(`Invalid replay segment "${line}"; expected 0-6 or 6-40 x4`);
    const start = Number(match[1]), end = Number(match[2]), speed = Number(match[3] ?? 1);
    if (![start, end, speed].every(Number.isFinite) || end <= start || speed <= 0) {
      throw new Error(`Invalid replay segment "${line}": end must exceed start and speed must be positive`);
    }
    return { start, end, speed };
  });
  return { file, segments };
}

// The proportional frame budget fits the complete cut to the measured chapter.
// Cumulative boundaries prevent rounding drift at the edit points.
export function fitReplaySegments(segments: ReplaySegment[], frames: number, fps: number) {
  if (!segments.length) return [];
  if (!Number.isInteger(frames) || frames < segments.length || !Number.isFinite(fps) || fps <= 0) {
    throw new Error("Replay chapter must have at least one frame per segment and positive fps");
  }
  const sourceSegments = segments.map(segment => {
    const trimBefore = Math.round(segment.start * fps);
    const trimAfter = Math.round(segment.end * fps);
    if ((segment.end - segment.start) * fps < 1 || trimAfter <= trimBefore) {
      throw new Error(`Replay segment ${segment.start}-${segment.end} spans less than one source frame at ${fps} fps`);
    }
    return { ...segment, trimBefore, trimAfter };
  });
  const weights = sourceSegments.map(s => (s.trimAfter - s.trimBefore) / fps / s.speed);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isFinite(total) || total <= 0) throw new Error("Replay cut duration must be finite and positive");
  let cursor = 0, elapsed = 0;
  return sourceSegments.map((segment, i) => {
    elapsed += weights[i]!;
    const proportionalEnd = Math.round(elapsed / total * frames);
    const earliestEnd = cursor + 1;
    const latestEnd = frames - (segments.length - i - 1);
    const clampedEnd = Math.min(latestEnd, Math.max(earliestEnd, proportionalEnd));
    const endFrame = i === segments.length - 1 ? frames : clampedEnd;
    const durationInFrames = endFrame - cursor;
    const from = cursor;
    cursor = endFrame;
    return { ...segment, from, durationInFrames, playbackRate: (segment.trimAfter - segment.trimBefore) / durationInFrames };
  });
}

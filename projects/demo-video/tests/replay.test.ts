import assert from "node:assert/strict";
import { test } from "node:test";
import { fitReplaySegments, parseReplay } from "../src/replay";
import { compileScript, PlanError } from "../scripts/plan";

test("replay accepts legacy clip and speed segments", () => {
  assert.deepEqual(parseReplay(["clip.mp4"]), { file: "clip.mp4", segments: [] });
  const spec = parseReplay(["clip.mp4", "0-6", "6-40 x4", "40-52"]);
  assert.deepEqual(spec.segments.map(s => s.speed), [1, 4, 1]);
  const cut = fitReplaySegments(spec.segments, 795, 30);
  assert.deepEqual(cut.map(s => s.durationInFrames), [180, 255, 360]);
  assert.deepEqual(cut.map(s => s.playbackRate), [1, 4, 1]);
  const fitted = fitReplaySegments(spec.segments, 900, 30);
  assert.equal(fitted.at(-1)!.from + fitted.at(-1)!.durationInFrames, 900);
  assert.deepEqual(fitReplaySegments([], 30, 30), []);
});

test("segment validation fails at plan time, including empty and malformed blocks", () => {
  for (const lines of [[], ["../clip.mp4"], ["clip.mp4", "6-0"], ["clip.mp4", "0-0"], ["clip.mp4", "0-6 x0"], ["clip.mp4", "0-6 xNaN"], ["clip.mp4", "garbage"]]) {
    assert.throws(() => parseReplay(lines));
    const script = `# Cinema\n## Cut\n<!-- visual: replay -->\n\n\`\`\`replay\n${lines.join("\n")}\n\`\`\`\n`;
    assert.throws(() => compileScript(script), PlanError);
  }
  assert.throws(() => fitReplaySegments([{ start: 0, end: 1, speed: 1 }], 0, 30));
});

test("fractional cut rates use the same rounded frames as video trims", () => {
  const segment = fitReplaySegments(parseReplay(["clip.mp4", "1.01-2.02"]).segments, 60, 30)[0]!;
  assert.equal(segment.trimBefore, 30);
  assert.equal(segment.trimAfter, 61);
  assert.equal(segment.playbackRate, 31 / 60);
  const tight = fitReplaySegments(parseReplay(["clip.mp4", "0-100", "100-101", "101-102"]).segments, 3, 30);
  assert.deepEqual(tight.map(s => s.durationInFrames), [1, 1, 1]);
});

test("sub-frame ranges fail clearly before rendering, naming the chapter", () => {
  for (const range of ["1.01-1.02", "1.01-1.04"]) {
    assert.throws(() => fitReplaySegments(parseReplay(["clip.mp4", range]).segments, 60, 30), /less than one source frame at 30 fps/);
    const script = `# Cinema\n## Fractional cut\n<!-- visual: replay -->\n\n\`\`\`replay\nclip.mp4\n${range}\n\`\`\`\n`;
    assert.throws(() => compileScript(script), /Chapter "Fractional cut": Replay segment .*less than one source frame/);
  }
});

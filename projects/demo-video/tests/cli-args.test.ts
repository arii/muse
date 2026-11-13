import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseArgs, USAGE } from "../scripts/cli";

describe("parseArgs", () => {
  it("parses command and positional script path", () => {
    const args = parseArgs(["render", "templates/demo-video.md"]);
    assert.equal(args.command, "render");
    assert.deepEqual(args.positional, ["templates/demo-video.md"]);
    assert.equal(args.error, undefined);
  });

  it("parses value flags", () => {
    const args = parseArgs([
      "render",
      "--script",
      "a.md",
      "--out",
      "out/x.mp4",
      "--concurrency",
      "2",
      "--browser-executable",
      "/bin/chrome",
    ]);
    assert.equal(args.flags["script"], "a.md");
    assert.equal(args.flags["out"], "out/x.mp4");
    assert.equal(args.flags["concurrency"], "2");
    assert.equal(args.flags["browser-executable"], "/bin/chrome");
  });

  it("parses boolean flags", () => {
    const args = parseArgs(["render", "a.md", "--silent", "--dry-run"]);
    assert.equal(args.flags["silent"], true);
    assert.equal(args.flags["dry-run"], true);
  });

  it("rejects unknown flags", () => {
    const args = parseArgs(["render", "a.md", "--wat"]);
    assert.match(args.error!, /unknown flag: --wat/);
  });

  it("rejects value flags missing their value", () => {
    const args = parseArgs(["render", "a.md", "--out"]);
    assert.match(args.error!, /--out requires a value/);
  });

  it("rejects unknown commands", () => {
    const args = parseArgs(["explode", "a.md"]);
    assert.match(args.error!, /unknown command: explode/);
  });

  it("recognizes --help", () => {
    assert.equal(parseArgs(["--help"]).help, true);
    assert.equal(parseArgs(["-h"]).help, true);
  });

  it("ships a usage block naming every command", () => {
    for (const word of ["plan", "tts", "render", "--silent", "--dry-run"]) {
      assert.ok(USAGE.includes(word), `usage missing ${word}`);
    }
  });
});

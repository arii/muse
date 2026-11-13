import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  checkFfmpeg,
  checkNode,
  formatPreflightReport,
  kokoroModelPresent,
  nodeVersionSatisfies,
  parseBrowserDownloadHint,
  resolveBrowserExecutable,
} from "../scripts/preflight";

const ENGINES = "^20.19.0 || ^22.13.0 || >=24";

describe("nodeVersionSatisfies", () => {
  it("accepts the sandbox node 20.20.2 against ^20.19.0", () => {
    assert.ok(nodeVersionSatisfies("v20.20.2", ENGINES));
  });

  it("accepts caret floors and >= clauses", () => {
    assert.ok(nodeVersionSatisfies("v22.13.0", ENGINES));
    assert.ok(nodeVersionSatisfies("v24.1.0", ENGINES));
  });

  it("rejects below-floor versions", () => {
    assert.ok(!nodeVersionSatisfies("v20.18.3", ENGINES));
    assert.ok(!nodeVersionSatisfies("v22.12.0", ENGINES));
    assert.ok(!nodeVersionSatisfies("v18.20.0", ENGINES));
  });
});

describe("checkNode", () => {
  it("passes with the version as detail", () => {
    const check = checkNode("v20.20.2", ENGINES);
    assert.equal(check.pass, true);
    assert.equal(check.detail, "v20.20.2");
  });

  it("fails with the engines range named", () => {
    const check = checkNode("v19.0.0", ENGINES);
    assert.equal(check.pass, false);
    assert.match(check.detail, /engines/);
  });
});

describe("checkFfmpeg", () => {
  it("extracts the version from ffmpeg output", () => {
    const check = checkFfmpeg(
      "ffmpeg version 7.1.5-0+deb13u1 Copyright (c) 2000-2024 the FFmpeg developers"
    );
    assert.equal(check.pass, true);
    assert.match(check.detail, /7\.1\.5/);
  });

  it("fails on empty output", () => {
    assert.equal(checkFfmpeg("").pass, false);
    assert.equal(checkFfmpeg("command not found").pass, false);
  });
});

describe("parseBrowserDownloadHint", () => {
  it("extracts a full chrome-for-testing URL", () => {
    const hint = parseBrowserDownloadHint(
      "Download failed: https://storage.googleapis.com/chrome-for-testing-public/131.0.6778.204/linux64/chrome-headless-shell-linux64.zip"
    );
    assert.match(hint.url!, /chrome-headless-shell-linux64\.zip$/);
    assert.equal(hint.version, undefined);
  });

  it("falls back to the printed build version", () => {
    const hint = parseBrowserDownloadHint(
      "Failed to download chrome-headless-shell build 131.0.6778.204"
    );
    assert.equal(hint.version, "131.0.6778.204");
  });

  it("returns nothing on unrelated output", () => {
    assert.deepEqual(parseBrowserDownloadHint("some timeout"), {});
  });
});

describe("resolveBrowserExecutable", () => {
  const files: Record<string, number> = {
    "/cache/a/chrome-headless-shell": 1000,
    "/cache/empty": 0,
  };
  const exists = (p: string) => p in files;
  const size = (p: string) => files[p] ?? 0;

  it("returns the first existing non-empty candidate", () => {
    assert.equal(
      resolveBrowserExecutable(exists, size, [
        "/cache/empty",
        "/cache/a/chrome-headless-shell",
      ]),
      "/cache/a/chrome-headless-shell"
    );
  });

  it("returns null when nothing is usable", () => {
    assert.equal(
      resolveBrowserExecutable(exists, size, ["/cache/empty", "/missing"]),
      null
    );
  });
});

describe("kokoroModelPresent", () => {
  it("accepts an onnx weights file in the kokoro cache", () => {
    assert.ok(
      kokoroModelPresent(
        ["onnx-community/Kokoro-82M-v1.0-ONNX/model.onnx", "voices.bin"],
        "Kokoro-82M-v1.0-ONNX"
      )
    );
  });

  it("rejects an unrelated or empty cache", () => {
    assert.ok(!kokoroModelPresent(["voices.bin"], "Kokoro-82M-v1.0-ONNX"));
    assert.ok(!kokoroModelPresent([], "Kokoro-82M-v1.0-ONNX"));
  });
});

describe("formatPreflightReport", () => {
  it("prints PASS/FAIL lines", () => {
    const report = formatPreflightReport([
      { name: "node", pass: true, detail: "v20.20.2" },
      { name: "browser", pass: false, detail: "missing" },
    ]);
    assert.equal(report, "PASS node: v20.20.2\nFAIL browser: missing");
  });
});

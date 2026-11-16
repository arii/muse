import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { serveDirectory } from "../scripts/capture-server";
import { captureBrowser, captureOptions, waitForCinema } from "../scripts/capture";

test("static server binds loopback and contains decoded paths and symlinks", async () => {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "cinema-server-"));
  const root = path.join(parent, "site");
  fs.mkdirSync(root);
  fs.writeFileSync(path.join(root, "index.html"), "cinema");
  fs.writeFileSync(path.join(parent, "private.txt"), "outside");
  fs.symlinkSync(path.join(parent, "private.txt"), path.join(root, "link.txt"));
  fs.symlinkSync(parent, path.join(root, "escape"));
  fs.mkdirSync(path.join(root, "nested"));
  fs.symlinkSync(path.join(parent, "private.txt"), path.join(root, "nested", "index.html"));
  const { server, url } = await serveDirectory(root);
  const request = (pathname: string) => new Promise<number>((resolve, reject) => {
    http.get(`${url}${pathname}`, response => { response.resume(); resolve(response.statusCode!); }).on("error", reject);
  });
  try {
    assert.equal((server.address() as { address: string }).address, "127.0.0.1");
    assert.equal(await (await fetch(url)).text(), "cinema");
    for (const pathname of ["/%2e%2e%2fprivate.txt", "/..%2fprivate.txt", "/link.txt", "/escape/private.txt", "/nested/"]) {
      assert.equal(await request(pathname), 403, pathname);
    }
    assert.equal(await request("/%ZZ"), 400);
    assert.equal(await request("/missing"), 404);
    assert.equal(await request("/index.html/missing"), 404);
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    fs.rmSync(parent, { recursive: true, force: true });
  }
});

test("static server logs unexpected filesystem failures and returns 500", async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "cinema-error-"));
  fs.writeFileSync(path.join(root, "index.html"), "cinema");
  const { server, url } = await serveDirectory(root);
  const failure = Object.assign(new Error("file descriptor limit"), { code: "EMFILE" });
  const createReadStream = fs.createReadStream;
  t.mock.method(fs, "createReadStream", () => {
    const stream = createReadStream(path.join(root, "index.html"));
    stream.destroy(failure);
    return stream;
  });
  const logged = t.mock.method(console, "error", () => {});
  try {
    const response = await fetch(url);
    assert.equal(response.status, 500);
    assert.equal(await response.text(), "Internal server error");
    assert.equal(logged.mock.calls[0]!.arguments[1], failure);
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("capture readiness errors retain the original message and cause", async () => {
  const cause = new Error("renderer crashed");
  await assert.rejects(waitForCinema({ waitForFunction: async () => { throw cause; } }, 500, "http://localhost/"), error => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /renderer crashed/);
    assert.equal(error.cause, cause);
    return true;
  });
});

test("browser selection prefers flag over environment", () => {
  const previous = process.env.CHROME_PATH;
  process.env.CHROME_PATH = "/environment/chrome";
  try {
    assert.equal(captureBrowser("/flag/chrome"), "/flag/chrome");
    assert.equal(captureBrowser(), "/environment/chrome");
  } finally {
    if (previous === undefined) delete process.env.CHROME_PATH;
    else process.env.CHROME_PATH = previous;
  }
});

test("capture defaults and invalid arguments", () => {
  const defaults = captureOptions(["--url", "http://localhost/", "--out", "clip.mp4"])!;
  assert.equal(defaults.fps, 30);
  assert.equal(defaults.duration, undefined);
  assert.equal(defaults.width, 1280);
  assert.equal(captureOptions(["--help"]), null);
  for (const extra of [["--fps", "0"], ["--size", "1279x720"], ["--duration", "NaN"], ["--stills", "-1"], ["--path", "//evil.test/"]]) {
    assert.throws(() => captureOptions(["--url", "http://localhost/", "--out", "clip.mp4", ...extra]));
  }
  assert.throws(() => captureOptions([]), /exactly one/);
});

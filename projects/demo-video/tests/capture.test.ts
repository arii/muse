import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { serveDirectory } from "../scripts/capture-server";
import { captureOptions } from "../scripts/capture";

test("static server binds loopback and contains decoded paths and symlinks", async () => {
  const parent = fs.mkdtempSync(path.join(os.tmpdir(), "cinema-server-"));
  const root = path.join(parent, "site");
  fs.mkdirSync(root);
  fs.writeFileSync(path.join(root, "index.html"), "cinema");
  fs.writeFileSync(path.join(parent, "private.txt"), "outside");
  fs.symlinkSync(path.join(parent, "private.txt"), path.join(root, "link.txt"));
  fs.symlinkSync(parent, path.join(root, "escape"));
  const { server, url } = await serveDirectory(root);
  const request = (pathname: string) => new Promise<number>((resolve, reject) => {
    http.get(`${url}${pathname}`, response => { response.resume(); resolve(response.statusCode!); }).on("error", reject);
  });
  try {
    assert.equal((server.address() as { address: string }).address, "127.0.0.1");
    assert.equal(await (await fetch(url)).text(), "cinema");
    for (const pathname of ["/%2e%2e%2fprivate.txt", "/..%2fprivate.txt", "/link.txt", "/escape/private.txt"]) {
      assert.equal(await request(pathname), 403, pathname);
    }
    assert.equal(await request("/%ZZ"), 400);
    assert.equal(await request("/missing"), 404);
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    fs.rmSync(parent, { recursive: true, force: true });
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

import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const TYPES: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".gz": "application/gzip", ".svg": "image/svg+xml",
  ".png": "image/png", ".jpg": "image/jpeg", ".woff2": "font/woff2", ".mp4": "video/mp4",
};

export async function serveDirectory(directory: string) {
  const root = fs.realpathSync(directory);
  if (!fs.statSync(root).isDirectory()) throw new Error(`Not a directory: ${root}`);
  const server = http.createServer((request, response) => {
    const allowPath = (file: string) => {
      if (file === root || file.startsWith(root + path.sep)) return true;
      response.writeHead(403).end("Outside static root");
      return false;
    };
    const fileError = (error: NodeJS.ErrnoException) => {
      const missing = error.code === "ENOENT" || error.code === "ENOTDIR";
      if (!missing) console.error("capture server:", error);
      if (response.headersSent) response.destroy();
      else response.writeHead(missing ? 404 : 500).end(missing ? "Not found" : "Internal server error");
    };
    let requested: string;
    try {
      requested = decodeURIComponent((request.url ?? "/").split("?")[0]!);
      if (requested.includes("\0")) throw new Error("NUL in URL");
    } catch {
      response.writeHead(400).end("Invalid URL");
      return;
    }
    let file = path.resolve(root, `.${requested}`);
    if (!allowPath(file)) return;
    try {
      file = fs.realpathSync(file);
      if (!allowPath(file)) return;
      if (fs.statSync(file).isDirectory()) file = fs.realpathSync(path.join(file, "index.html"));
      if (!allowPath(file)) return;
      if (!fs.statSync(file).isFile()) {
        response.writeHead(404).end("Not found");
        return;
      }
      const stream = fs.createReadStream(file);
      stream.on("error", fileError);
      response.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
      stream.pipe(response);
    } catch (error) {
      fileError(error as NodeJS.ErrnoException);
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address() as { port: number };
  return { server, url: `http://127.0.0.1:${address.port}` };
}

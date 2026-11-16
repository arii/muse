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
  const contained = (file: string) => file === root || file.startsWith(root + path.sep);
  const server = http.createServer((request, response) => {
    let requested: string;
    try {
      requested = decodeURIComponent((request.url ?? "/").split("?")[0]!);
      if (requested.includes("\0")) throw new Error("NUL in URL");
    } catch {
      response.writeHead(400).end("Invalid URL");
      return;
    }
    let file = path.resolve(root, `.${requested}`);
    if (!contained(file)) {
      response.writeHead(403).end("Outside static root");
      return;
    }
    try {
      file = fs.realpathSync(file);
      if (!contained(file)) {
        response.writeHead(403).end("Outside static root");
        return;
      }
      if (fs.statSync(file).isDirectory()) file = fs.realpathSync(path.join(file, "index.html"));
      if (!contained(file)) {
        response.writeHead(403).end("Outside static root");
        return;
      }
      if (!fs.statSync(file).isFile()) throw new Error("Not a file");
      response.writeHead(200, { "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream" });
      const stream = fs.createReadStream(file);
      stream.on("error", () => response.destroy());
      stream.pipe(response);
    } catch {
      response.writeHead(404).end("Not found");
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address() as { port: number };
  return { server, url: `http://127.0.0.1:${address.port}` };
}

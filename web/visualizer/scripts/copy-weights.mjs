// Copies the exported model weights into public/weights/ so Vite serves them
// as static assets. Runs automatically before `dev`/`build` (see package.json).
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "weights");
const dest = join(here, "..", "public", "weights");

mkdirSync(dest, { recursive: true });
for (const entry of readdirSync(src)) {
  if (entry === ".gitkeep") continue;
  cpSync(join(src, entry), join(dest, entry));
}

console.log(`Copied weights from ${src} to ${dest}`);

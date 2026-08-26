// Download the MediaPipe FaceLandmarker runtime, its model and their licence
// into public/, so the deployed site serves everything from its own origin.
// Nothing is fetched from a third party at run time, which is what lets the
// Content-Security-Policy stay 'self'-only and keeps the demo self-contained.
//
//   node scripts/fetch-assets.mjs
//
// The downloaded files are gitignored (large binaries, not source). Versions
// and expected SHA-256 hashes live in assets-manifest.mjs; every download is
// verified against its hash before it is written, so a substituted response
// from a CDN can never be vendored.

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { FILES, TASKS_VERSION } from "./assets-manifest.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(HERE, "..", "public");

async function download(url, rel, sha256) {
  const dest = join(PUBLIC, rel);
  await mkdir(dirname(dest), { recursive: true });
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} fetching ${url}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== sha256) {
    throw new Error(`integrity mismatch for ${rel}: expected ${sha256}, got ${digest}`);
  }
  await writeFile(dest, bytes);
  console.log(`  ${rel}  (${bytes.length.toLocaleString()} bytes, sha256 verified)`);
}

console.log(`Fetching MediaPipe tasks-vision ${TASKS_VERSION}, the model and the licence...`);
for (const [url, rel, sha256] of FILES) {
  await download(url, rel, sha256);
}
console.log("Done. The demo now runs fully from its own origin.");

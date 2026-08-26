// Predeploy guard. The MediaPipe runtime, model and licence are vendored into
// public/ by fetch-assets.mjs but gitignored, so a fresh clone can deploy a
// site that looks complete and cannot start. Firebase Hosting uploads public/
// as a full snapshot without validating it, so this check fails the deploy
// instead, and it verifies each file's SHA-256 against assets-manifest.mjs, so
// a deploy also proves the vendored bytes are the pinned ones.
//
// Wired up in firebase.json as hosting.predeploy.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { FILES } from "./assets-manifest.mjs";

const PUBLIC = join(dirname(fileURLToPath(import.meta.url)), "..", "public");

let ok = true;
for (const [, rel, sha256] of FILES) {
  try {
    const digest = createHash("sha256").update(readFileSync(join(PUBLIC, rel))).digest("hex");
    if (digest !== sha256) {
      console.error(`check-assets: ${rel} does not match its pinned sha256`);
      ok = false;
    }
  } catch {
    console.error(`check-assets: missing ${rel}`);
    ok = false;
  }
}

if (!ok) {
  console.error("check-assets: run `node scripts/fetch-assets.mjs` first.");
  process.exit(1);
}
console.log("check-assets: all vendored assets present and integrity-checked.");

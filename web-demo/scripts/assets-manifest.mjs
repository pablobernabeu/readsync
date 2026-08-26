// The vendored third-party assets, pinned by version and by content.
//
// The version is tasks-vision 0.10.35, the release the desktop toolkit was
// developed and tested against (pyproject.toml bounds mediapipe to the same
// 0.10 series); bump both together, never one side alone. Every entry also
// carries the SHA-256 of the expected bytes, so a substituted download can
// never be vendored or deployed: fetch-assets.mjs verifies on download and
// check-assets.mjs re-verifies before every deploy. When bumping the version,
// refresh the hashes from a download you have inspected.
//
// The runtime and the model are Apache-2.0, so their licence text is vendored
// alongside them and served with the site.

export const TASKS_VERSION = "0.10.35";

const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/" +
  "face_landmarker/float16/1/face_landmarker.task";

// [url, path under public/, sha256 of the expected bytes]
export const FILES = [
  [
    `${CDN}/vision_bundle.mjs`,
    "vendor/vision_bundle.mjs",
    "55d7ab624fbb70dcc5adc4ae6d7ea9cfcb569139d3dbfbf2b1deafcb966bc0fe",
  ],
  [
    `${CDN}/wasm/vision_wasm_internal.js`,
    "vendor/wasm/vision_wasm_internal.js",
    "e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c",
  ],
  [
    `${CDN}/wasm/vision_wasm_internal.wasm`,
    "vendor/wasm/vision_wasm_internal.wasm",
    "6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc",
  ],
  [
    `${CDN}/wasm/vision_wasm_nosimd_internal.js`,
    "vendor/wasm/vision_wasm_nosimd_internal.js",
    "438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296",
  ],
  [
    `${CDN}/wasm/vision_wasm_nosimd_internal.wasm`,
    "vendor/wasm/vision_wasm_nosimd_internal.wasm",
    "8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31",
  ],
  [
    MODEL_URL,
    "models/face_landmarker.task",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
  ],
  [
    // The canonical Apache-2.0 text; the npm package declares the licence but
    // ships no licence file.
    "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "vendor/LICENSE",
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
  ],
];

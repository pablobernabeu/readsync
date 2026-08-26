# Interlexis webcam reading demo (browser)

A fully client-side, head-pose-aware webcam gaze demo for reading, deployed at
https://interlexis.web.app. It is the browser benchmark strand of the Interlexis
reading project: the same gaze pipeline as the desktop `readsync` toolkit,
ported to JavaScript and run entirely in the browser.

## Security model

This is the point of the demo, so it is built in from the start.

- **Nothing is uploaded.** Camera frames are processed in the browser by MediaPipe
  FaceLandmarker. The gaze estimates, the calibration and the event log never
  leave the device. The only way data leaves is if you click *Download CSV*, which
  writes `interlexis-session.csv` locally.
- **No third party at run time.** The MediaPipe runtime, the model and the stimuli
  are all served from the site's own origin, vendored by `scripts/fetch-assets.mjs`
  and pinned by SHA-256 in `scripts/assets-manifest.mjs`, so a substituted download
  can neither be vendored nor deployed. The Content-Security-Policy in
  `firebase.json` locks every source to the site's own origin, with only the
  narrow additions the pipeline needs, `'wasm-unsafe-eval'` for MediaPipe's
  WebAssembly and `blob:`/`data:` for the worker, images and media the browser
  creates locally. It has no `connect-src` to any other host, so the page cannot
  exfiltrate data even accidentally.
- **Least privilege.** `Permissions-Policy` grants the camera only, and disables
  microphone and geolocation. Framing is denied. No analytics, cookies or trackers
  are used.

The demo takes the same offline-first, secure-by-default stance as the desktop
toolkit, expressed for the browser.

## Prescreen instrument (planned build)

This demo is the benchmark strand: the camera is on and nothing is uploaded.
The research programme's national prescreen is a separate, camera-free
build, frozen by the research software engineer before the recruitment funnel
opens. It will deliver the validated adult self-report questionnaire, the adult
checklist, LexTALE and a brief reading task, and submit scored responses only,
no raw keystrokes and no camera access, under the participant's consent. Because
submission is a same-origin request (a hosting rewrite to a function on the same
project keeps the `connect-src 'self'` policy intact), the no-third-party stance
of this site carries over. The enumerated response fields feed the project's
data-protection impact assessment and retention schedule.

## Run it locally

```bash
cd web-demo
node scripts/fetch-assets.mjs        # vendor the MediaPipe runtime, model and licence (once)
npm test                             # gaze-maths checks, including a fixture generated from the desktop implementation
firebase emulators:start --only hosting   # or: firebase serve --only hosting
```

Then open the printed `http://localhost` URL. `localhost` is a secure context, so
the browser allows camera access there. Any static server works, but it must serve
`.mjs` as JavaScript and `.wasm` as `application/wasm`. The Firebase emulator and
`firebase serve` do this and also apply the security headers, so they are the best
way to test.

## Deploy to Firebase Hosting

This is a static-only Hosting site: no Cloud Functions, no database, no
server-side data.

```bash
cd web-demo
node scripts/fetch-assets.mjs        # ensure vendored assets are present
firebase deploy --only hosting
```

A predeploy hook (`scripts/check-assets.mjs`) fails the deploy if any vendored
file is missing or does not match its pinned SHA-256, because those files are
gitignored and a fresh clone would otherwise deploy a site that cannot start. HTML, scripts and
stimuli are served with `Cache-Control: no-cache`, so a redeploy shows up on a
normal reload; the vendored runtime and model are immutable, version-pinned
binaries and are cached for a year.

## What it does

1. Asks for camera access and loads the model. The camera is requested last, so
   the permission prompt appears only after the model and stimuli have loaded.
2. Calibrates at two head positions (a 3x3 grid each), so the model can learn to
   compensate for head pose, and reports a leave-one-out accuracy estimate, with a
   redo option if it is poor. The webcam self-view is hidden while the dots are
   shown, so it cannot cover the corner targets.
3. Shows controlled frequency sentences with a live gaze dot and the word under
   gaze. The demo presents one member of each matched pair, alternating condition
   across pairs, so the reader never sees the same frame twice
   (`public/js/stimuli.js`). Every stage is operable by button as well as by
   keyboard, so touch devices work. The dot turns amber and asks for a
   recalibration once the head moves beyond the calibrated range or the window is
   resized, blinks hold the last gaze, and recalibrating resumes at the
   interrupted sentence.
4. Lets you download the in-browser event log as a CSV with the columns
   `type,t,passage,word,x,y,valid` and the event types `session_start`,
   `passage_onset`, `gaze`, `word_enter`, `passage_offset`, `recalibrate` and
   `session_end`. All timestamps share one session clock.

## Limits

Single-camera webcam gaze is coarse: it resolves a region of a line, never a
single letter. The demo serves the methods and screening strand and cannot
replace the research-grade infrared tracker used for word-level measures.

## Layout

```
web-demo/
  public/
    index.html, styles.css, favicon.svg
    js/linalg.js    linear algebra (port of readsync._linalg)
    js/gaze.js      gaze maths (port of readsync.webcam tracker + gaze)
    js/stimuli.js   demo stimulus selection (one member per pair)
    js/app.js       camera, calibration and reading orchestration
    stimuli/        the controlled frequency set
    vendor/, models/   vendored MediaPipe runtime and model (fetched, gitignored)
  scripts/assets-manifest.mjs   pinned versions and SHA-256 hashes of the vendored assets
  scripts/fetch-assets.mjs   vendor the runtime, model and licence, verifying each hash
  scripts/check-assets.mjs   predeploy guard re-verifying the vendored assets
  test/gaze.test.mjs   Node checks of the gaze maths, against a desktop-generated fixture
  firebase.json, .firebaserc
```

## A note on versions and the Content-Security-Policy

The MediaPipe runtime is pinned to tasks-vision 0.10.35, with per-file SHA-256
hashes, in `assets-manifest.mjs`; it is the release the desktop toolkit was
developed against, so bump both sides together and refresh the hashes from a
download you have inspected. The runtime and the model are Apache-2.0, and
their licence is vendored and served alongside them at `vendor/LICENSE`.
If a future MediaPipe build reports
a blocked `eval`, the minimal change is to widen `script-src` for that build. Do
not add any `connect-src` host, since that is what guarantees no data can leave.

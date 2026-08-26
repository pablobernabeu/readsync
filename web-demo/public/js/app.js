// Browser orchestration for the Interlexis webcam reading demo. Everything runs
// on this device: the camera frames go to MediaPipe FaceLandmarker in the
// browser, the gaze maths (gaze.js) is the same as the desktop readsync
// toolkit, and the event log is held in memory and only ever downloaded
// locally. Nothing is uploaded.

import { FaceLandmarker, FilesetResolver } from "/vendor/vision_bundle.mjs";
import {
  HeadAwareModel,
  crossValidatedError,
  medianSignal,
  signalsFromResult,
} from "/js/gaze.js";
import { selectDemoItems } from "/js/stimuli.js";

const CALIB_AXIS = [0.12, 0.5, 0.88]; // a 3x3 grid per pose
const POSES = 2; // calibrate at two head positions so head pose is learned
const SAMPLES_PER_TARGET = 18;
const SETTLE_MS = 700;
const BLINK_OPENNESS = 0.15;
const FX_RANGE = [-0.6, 1.6];
const FY_RANGE = [-1.6, 1.6];
const EMA_ALPHA = 0.35;
const DRIFT_LIMIT_DEG = 12;
const MAX_CV_ERROR = 0.06; // accept if leave-one-out error is within 6% of screen
const RIDGE = 1e-3;
const FACE_LOST_FRAMES = 45; // consecutive frames without a usable signal; ~0.75 s at 60 Hz
const RESIZE_TOLERANCE = 0.1; // fraction the window may change before a warning
const CALIB_HINT = "Follow the dot with your eyes.";

const $ = (id) => document.getElementById(id);
const screens = ["intro", "instruction", "confirm", "reading", "done"];
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const video = $("cam");
const hint = $("hint");

let landmarker = null;
let currentSignal = null;
let frameWaiters = [];
let detectLoopStarted = false;
let sessionClock = 0;
const events = [];

const now = () => (performance.now() - sessionClock) / 1000;

// Show one screen panel, or none at all when name is null. Calibration passes
// null so that only the full-screen dots are visible: showing a panel would
// leave an empty grey box in the middle of the screen behind the targets.
function showScreen(name) {
  for (const id of screens) $(id).hidden = id !== name;
}

function setHint(text) {
  hint.textContent = text;
}

function sizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(window.innerWidth * dpr);
  overlay.height = Math.round(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", sizeOverlay);

function clearOverlay() {
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
}

function nextFrame() {
  return new Promise((resolve) => frameWaiters.push(resolve));
}

function detectLoop() {
  if (landmarker && video.readyState >= 2) {
    try {
      const result = landmarker.detectForVideo(video, performance.now());
      currentSignal = signalsFromResult(result);
    } catch {
      // One failed detection, a non-monotonic timestamp after a tab suspend
      // for example, counts as a missed frame and must not kill the loop.
      currentSignal = null;
    }
    const waiters = frameWaiters;
    frameWaiters = [];
    for (const resolve of waiters) resolve(currentSignal);
  }
  requestAnimationFrame(detectLoop);
}

function usable(signal) {
  if (!signal || signal.openness < BLINK_OPENNESS) return false;
  return (
    signal.fx >= FX_RANGE[0] && signal.fx <= FX_RANGE[1] &&
    signal.fy >= FY_RANGE[0] && signal.fy <= FY_RANGE[1]
  );
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// Resolve when a button is clicked or its shortcut key is pressed, whichever
// comes first. Clicking is the primary path; the keys stay as an optional
// shortcut. Each option is { el, value, keys }, where keys are lowercase and
// " " is the space bar. Key events are ignored while one of the option buttons
// itself has focus, so that pressing Space on a focused button activates that
// button natively instead of being hijacked by the window-level shortcut.
// Every listener is removed before resolving, so screens never stack them.
function waitAction(options) {
  return new Promise((resolve) => {
    const cleanups = [];
    const finish = (value) => {
      while (cleanups.length) cleanups.pop()();
      resolve(value);
    };
    for (const opt of options) {
      const onClick = () => finish(opt.value);
      opt.el.addEventListener("click", onClick);
      cleanups.push(() => opt.el.removeEventListener("click", onClick));
    }
    const onKey = (e) => {
      if (e.repeat) return;
      if (options.some((o) => o.el === e.target)) return;
      const key = e.key === " " ? " " : e.key.toLowerCase();
      const match = options.find((o) => o.keys && o.keys.includes(key));
      if (match) {
        e.preventDefault();
        finish(match.value);
      }
    };
    window.addEventListener("keydown", onKey);
    cleanups.push(() => window.removeEventListener("keydown", onKey));
  });
}

function drawTarget(nx, ny, progress) {
  clearOverlay();
  const x = nx * window.innerWidth;
  const y = ny * window.innerHeight;
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(x, y, 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#cc0000";
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#00b35a";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(x, y, 22, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  ctx.stroke();
}

function drawDot(nx, ny, valid) {
  const x = clamp01(nx) * window.innerWidth;
  const y = clamp01(ny) * window.innerHeight;
  ctx.fillStyle = valid ? "#00e676" : "#ffa500";
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, Math.PI * 2);
  ctx.fill();
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  video.classList.add("show");
}

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks("/vendor/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "/models/face_landmarker.task" },
    numFaces: 1,
    runningMode: "VIDEO",
    outputFacialTransformationMatrixes: true,
  });
}

async function collectTarget(nx, ny, viewport) {
  const samples = [];
  const settleUntil = performance.now() + SETTLE_MS;
  let missed = 0;
  while (samples.length < SAMPLES_PER_TARGET) {
    if (resizedFrom(viewport)) return null; // the caller restarts calibration
    const signal = await nextFrame();
    if (usable(signal) && performance.now() > settleUntil) {
      samples.push(signal);
      if (missed >= FACE_LOST_FRAMES) setHint(CALIB_HINT);
      missed = 0;
    } else {
      missed += 1;
      if (missed === FACE_LOST_FRAMES) {
        // No panel is on screen during collection, so the footer carries this.
        setHint(
          "Your eyes are not being found. Check the lighting and framing; " +
          "collection resumes as soon as they are found."
        );
      }
    }
    drawTarget(nx, ny, samples.length / SAMPLES_PER_TARGET);
  }
  return samples;
}

function viewportNow() {
  return { w: window.innerWidth, h: window.innerHeight };
}

function resizedFrom(viewport) {
  return (
    Math.abs(window.innerWidth - viewport.w) / viewport.w > RESIZE_TOLERANCE ||
    Math.abs(window.innerHeight - viewport.h) / viewport.h > RESIZE_TOLERANCE
  );
}

async function runCalibration() {
  $("reading-controls").hidden = true;
  // Targets are drawn at fractions of the live window, so every pose must be
  // collected at one window size. A resize beyond tolerance mid-calibration
  // restarts the collection at the new size; the snapshot returned here is the
  // one the reading-phase guard measures against.
  for (;;) {
    const viewport = viewportNow();
    const signals = [];
    const targets = [];
    const groups = [];
    let resized = false;
    for (let pose = 0; pose < POSES && !resized; pose++) {
      $("instruction-text").innerHTML =
        pose === 0
          ? "<h2>Calibration pose 1</h2><p>Sit the way you will when reading, look " +
            "straight at the screen, and follow each dot with your eyes. Keep your head " +
            "still through this pose.</p>"
          : `<h2>Calibration pose ${pose + 1}</h2><p>Now turn your head to one side, or ` +
            "lean in or back, and hold that clearly different position while you follow " +
            "the dots. A noticeably different pose here, not a tiny shift, is what lets " +
            "the tracker learn to follow your head.</p>";
      showScreen("instruction");
      setHint("Click Begin when you are ready.");
      await waitAction([{ el: $("instruction-go"), value: "go", keys: [" "] }]);
      showScreen(null); // dots only: any panel would sit behind them as an empty box
      $("passage").innerHTML = "";
      $("reading-note").textContent = "";
      setHint(CALIB_HINT);
      video.classList.remove("show"); // the self-view would cover the corner target
      let group = 0;
      for (const ny of CALIB_AXIS) {
        for (const nx of CALIB_AXIS) {
          const samples = await collectTarget(nx, ny, viewport);
          if (samples === null) {
            resized = true;
            break;
          }
          signals.push(medianSignal(samples));
          targets.push([nx, ny]);
          groups.push(group++);
        }
        if (resized) break;
      }
      video.classList.add("show");
      clearOverlay(); // do not leave the last target on screen between poses
    }
    if (!resized) return { signals, targets, groups, viewport };
    setHint("The window was resized during calibration, so it restarts at the new size.");
  }
}

async function confirmCalibration(cvError) {
  const percent = (cvError * 100).toFixed(1);
  const good = cvError <= MAX_CV_ERROR;
  $("confirm-text").innerHTML =
    `<h2>Calibration error: ${percent}% of the screen</h2>` +
    `<p>${good ? "That looks good." : "That is on the high side; more light and a " +
      "steadier head will help."}</p>`;
  showScreen("confirm");
  setHint("");
  const choice = await waitAction([
    { el: $("confirm-use"), value: "use", keys: [" "] },
    { el: $("confirm-redo"), value: "redo", keys: ["r"] },
  ]);
  return choice === "use";
}

function renderPassage(text) {
  const passage = $("passage");
  passage.innerHTML = "";
  for (const token of text.split(/\s+/)) {
    const span = document.createElement("span");
    span.className = "word";
    span.textContent = token;
    passage.appendChild(span);
    passage.appendChild(document.createTextNode(" "));
  }
  return [...passage.querySelectorAll(".word")];
}

function wordAt(words, x, y) {
  for (let i = 0; i < words.length; i++) {
    const r = words[i].getBoundingClientRect();
    if (x >= r.left && x < r.right && y >= r.top && y < r.bottom) return i;
  }
  return -1;
}

// Reads passages with a live gaze dot. Returns { status, index } where status
// is "done", "quit" or "recalibrate" and index is the passage to resume at, so
// that a recalibration continues from the interrupted passage instead of
// starting over.
async function runReading(passages, model, startIndex, viewport) {
  let control = null;
  const onKey = (e) => {
    if (e.repeat) return;
    const k = e.key === " " ? " " : e.key.toLowerCase();
    if (k === " ") {
      e.preventDefault();
      control = "next";
    } else if (k === "c") control = "recalibrate";
    else if (k === "escape") control = "quit";
  };
  window.addEventListener("keydown", onKey);
  const controls = [
    { el: $("read-next"), value: "next" },
    { el: $("read-recal"), value: "recalibrate" },
    { el: $("read-finish"), value: "quit" },
  ];
  const clickCleanups = controls.map(({ el, value }) => {
    const onClick = () => { control = value; };
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  });
  showScreen("reading");
  $("reading-controls").hidden = false;
  let smoothed = null;
  let idx = startIndex;
  try {
    for (; idx < passages.length; idx++) {
      const passage = passages[idx];
      const words = renderPassage(passage.text);
      let currentWord = -1;
      control = null;
      events.push({ type: "passage_onset", t: now(), passage: passage.id });
      while (!control) {
        const signal = await nextFrame();
        clearOverlay();
        let highlighted = -1;
        let note = "";
        const t = now();
        const resized = resizedFrom(viewport);
        if (!usable(signal)) {
          note = signal ? "blink" : "searching for your eyes";
          if (smoothed) drawDot(smoothed[0], smoothed[1], false);
        } else {
          const [nx, ny] = model.map(signal);
          if (Number.isFinite(nx) && Number.isFinite(ny)) {
            smoothed = smoothed
              ? [EMA_ALPHA * nx + (1 - EMA_ALPHA) * smoothed[0], EMA_ALPHA * ny + (1 - EMA_ALPHA) * smoothed[1]]
              : [nx, ny];
            const drift = model.poseDriftDegrees(signal);
            const valid = drift <= DRIFT_LIMIT_DEG && !resized;
            const gx = clamp01(smoothed[0]) * window.innerWidth;
            const gy = clamp01(smoothed[1]) * window.innerHeight;
            events.push({ type: "gaze", t, x: Math.round(gx), y: Math.round(gy), valid });
            if (valid) {
              highlighted = wordAt(words, gx, gy);
              if (highlighted >= 0 && highlighted !== currentWord) {
                events.push({ type: "word_enter", t, passage: passage.id, word: highlighted });
                currentWord = highlighted;
              }
            } else if (resized) {
              note = "the window was resized since calibration; recalibrate";
            } else {
              note = "you have moved; recalibrate";
            }
            drawDot(smoothed[0], smoothed[1], valid);
          } else {
            note = "tracking lost";
          }
        }
        words.forEach((w, i) => w.classList.toggle("hit", i === highlighted));
        $("reading-note").textContent = note;
      }
      if (control === "recalibrate") return { status: "recalibrate", index: idx };
      events.push({ type: "passage_offset", t: now(), passage: passage.id });
      if (control === "quit") return { status: "quit", index: idx };
    }
    return { status: "done", index: passages.length };
  } finally {
    window.removeEventListener("keydown", onKey);
    clickCleanups.forEach((fn) => fn());
    $("reading-controls").hidden = true;
    clearOverlay();
  }
}

function downloadCsv() {
  const cols = ["type", "t", "passage", "word", "x", "y", "valid"];
  const lines = [cols.join(",")];
  for (const e of events) {
    lines.push(cols.map((c) => (e[c] === undefined ? "" : e[c])).join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "interlexis-session.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function loadPassages() {
  const response = await fetch("/stimuli/controlled_freq_en.json");
  if (!response.ok) {
    throw new Error(`the stimuli could not be loaded (HTTP ${response.status})`);
  }
  const data = await response.json();
  return selectDemoItems(data.items).map((it) => ({ id: it.id, text: it.text }));
}

function startupErrorMessage(err) {
  if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
    return (
      "Camera access was declined. The demo estimates gaze from the camera on " +
      "this device only; no video is recorded or uploaded. Allow camera access " +
      "in the browser and try again."
    );
  }
  if (err && err.name === "NotFoundError") {
    return "No camera was found on this device.";
  }
  return (
    "Could not start: " + (err && err.message ? err.message : err) + ". The " +
    "MediaPipe assets and model must be present (run fetch-assets), and camera " +
    "access must be allowed."
  );
}

async function run() {
  const startBtn = $("start");
  startBtn.disabled = true;
  startBtn.textContent = "Loading...";
  let passages;
  try {
    setHint("Loading the gaze model...");
    await initLandmarker();
    passages = await loadPassages();
    setHint("Waiting for camera permission...");
    await startCamera();
  } catch (err) {
    const note = $("intro-error");
    note.hidden = false;
    note.textContent = startupErrorMessage(err);
    startBtn.disabled = false;
    startBtn.textContent = "Allow camera and begin";
    setHint("Click the button to allow camera access and begin.");
    showScreen("intro");
    return;
  }
  if (!detectLoopStarted) {
    detectLoopStarted = true;
    detectLoop();
  }
  sessionClock = performance.now();
  events.length = 0;
  events.push({ type: "session_start", t: 0 });
  let index = 0;
  while (true) {
    const { signals, targets, groups, viewport } = await runCalibration();
    const model = HeadAwareModel.fit(signals, targets, RIDGE);
    const cvError = crossValidatedError(signals, targets, RIDGE, groups);
    if (!(await confirmCalibration(cvError))) continue;
    setHint("Next sentence: button or SPACE. Recalibrate: button or C. Finish: button or Esc.");
    const { status, index: reached } = await runReading(passages, model, index, viewport);
    if (status === "recalibrate") {
      index = reached;
      events.push({ type: "recalibrate", t: now() });
      continue;
    }
    break;
  }
  events.push({ type: "session_end", t: now() });
  const gaze = events.filter((e) => e.type === "gaze");
  const trusted = gaze.filter((e) => e.valid);
  $("done-text").innerHTML =
    `<h2>Done</h2><p>Recorded ${events.length} events in your browser ` +
    `(${gaze.length} gaze, ${trusted.length} trusted). Nothing was uploaded. ` +
    "You can download the log as a CSV, which is created and saved locally.</p>";
  showScreen("done");
  setHint("");
}

$("start").addEventListener("click", run);
$("download").addEventListener("click", downloadCsv);
$("restart").addEventListener("click", () => window.location.reload());
sizeOverlay();
setHint("Click the button to allow camera access and begin.");

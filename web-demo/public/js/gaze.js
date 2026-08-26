// Head-pose-aware webcam gaze, ported from the readsync Python pipeline
// (webcam/tracker.py and webcam/gaze.py). Pure functions plus a small model
// class, with no browser or MediaPipe dependency, so they are unit-tested in
// Node and reused unchanged in the browser. The MediaPipe FaceLandmarker is
// driven in app.js, which passes its result objects to signalsFromResult here.

import { ridgeFit } from "./linalg.js";

// Landmark indices in the 478-point model: (iris, corner, corner, upper lid,
// lower lid) for each eye.
export const RIGHT_EYE = [468, 33, 133, 159, 145];
export const LEFT_EYE = [473, 362, 263, 386, 374];

export function eyeFeature(landmarks, spec) {
  // Iris position in the eye's own frame, as fractions of width and height. Using
  // the eye's own axes makes the feature invariant to head roll.
  const [iris, a, b, up, lo] = spec;
  const ax = landmarks[a].x, ay = landmarks[a].y;
  const bx = landmarks[b].x, by = landmarks[b].y;
  const ix = landmarks[iris].x, iy = landmarks[iris].y;
  const axisX = bx - ax, axisY = by - ay;
  const width = Math.hypot(axisX, axisY) || 1e-6;
  const ux = axisX / width, uy = axisY / width;
  const fx = ((ix - ax) * ux + (iy - ay) * uy) / width;
  const cx = (ax + bx) / 2, cy = (ay + by) / 2;
  const height =
    Math.hypot(landmarks[lo].x - landmarks[up].x, landmarks[lo].y - landmarks[up].y) || 1e-6;
  const fy = ((ix - cx) * -uy + (iy - cy) * ux) / height;
  return [fx, fy];
}

export function eyeOpenness(landmarks, spec) {
  // Eye-aspect ratio (lid gap over eye width). Falls towards zero during a blink.
  const [, a, b, up, lo] = spec;
  const width =
    Math.hypot(landmarks[b].x - landmarks[a].x, landmarks[b].y - landmarks[a].y) || 1e-6;
  const height = Math.hypot(landmarks[lo].x - landmarks[up].x, landmarks[lo].y - landmarks[up].y);
  return height / width;
}

export function headPoseFromMatrix(data) {
  // Extract (yaw, pitch, roll) in radians from a 16-element 4x4 head-pose matrix
  // (FaceLandmarker facial transformation matrix, row-major). The exact convention
  // matters only for naming; the angles serve as head-orientation inputs and to
  // measure drift.
  const r = (i, j) => data[i * 4 + j];
  const sy = Math.hypot(r(0, 0), r(1, 0));
  if (sy > 1e-6) {
    return {
      roll: Math.atan2(r(1, 0), r(0, 0)),
      yaw: Math.atan2(-r(2, 0), sy),
      pitch: Math.atan2(r(2, 1), r(2, 2)),
    };
  }
  return { roll: 0, yaw: Math.atan2(-r(2, 0), sy), pitch: Math.atan2(-r(1, 2), r(1, 1)) };
}

export function signalsFromResult(result) {
  // Build a per-frame signal from a FaceLandmarker result, or null if no face.
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) return null;
  const lm = result.faceLandmarks[0];
  const [rfx, rfy] = eyeFeature(lm, RIGHT_EYE);
  const [lfx, lfy] = eyeFeature(lm, LEFT_EYE);
  const openness = (eyeOpenness(lm, RIGHT_EYE) + eyeOpenness(lm, LEFT_EYE)) / 2;
  let yaw = 0, pitch = 0, roll = 0;
  const mats = result.facialTransformationMatrixes;
  if (mats && mats.length) ({ yaw, pitch, roll } = headPoseFromMatrix(mats[0].data));
  return { fx: (rfx + lfx) / 2, fy: (rfy + lfy) / 2, yaw, pitch, roll, openness };
}

// Twelve design terms: a second-order polynomial in the iris feature, linear head
// yaw and pitch, and the products of head pose with the iris feature. The pose-by-
// feature products let head pose change the feature's gain, not only add an offset.
// That is what a still-head calibration held at two poses needs in order to fit
// both poses instead of averaging them. Roll is handled by the roll-invariant feature.
function design(s) {
  const { fx, fy, yaw, pitch } = s;
  return [
    1, fx, fy, fx * fx, fy * fy, fx * fy, yaw, pitch,
    yaw * fx, yaw * fy, pitch * fx, pitch * fy,
  ];
}

function median(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const n = ordered.length, mid = n >> 1;
  return n % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
}

export function medianSignal(samples) {
  const field = (k) => median(samples.map((s) => s[k]));
  return {
    fx: field("fx"), fy: field("fy"), yaw: field("yaw"),
    pitch: field("pitch"), roll: field("roll"), openness: field("openness"),
  };
}

export class HeadAwareModel {
  constructor(coeffsX, coeffsY, poseLo, poseHi) {
    this.coeffsX = coeffsX;
    this.coeffsY = coeffsY;
    this.poseLo = poseLo;
    this.poseHi = poseHi;
  }

  static fit(signals, targets, ridge = 1e-3) {
    if (signals.length !== targets.length) {
      throw new Error("signals and targets must have the same length");
    }
    const X = signals.map(design);
    const coeffsX = ridgeFit(X, targets.map((t) => t[0]), ridge);
    const coeffsY = ridgeFit(X, targets.map((t) => t[1]), ridge);
    const axis = (k) => signals.map((s) => s[k]);
    const poseLo = [Math.min(...axis("yaw")), Math.min(...axis("pitch")), Math.min(...axis("roll"))];
    const poseHi = [Math.max(...axis("yaw")), Math.max(...axis("pitch")), Math.max(...axis("roll"))];
    return new HeadAwareModel(coeffsX, coeffsY, poseLo, poseHi);
  }

  map(s) {
    const t = design(s);
    const nx = this.coeffsX.reduce((acc, c, i) => acc + c * t[i], 0);
    const ny = this.coeffsY.reduce((acc, c, i) => acc + c * t[i], 0);
    return [nx, ny];
  }

  poseDriftDegrees(s) {
    const outside = (v, lo, hi) => Math.max(0, lo - v, v - hi);
    const worst = Math.max(
      outside(s.yaw, this.poseLo[0], this.poseHi[0]),
      outside(s.pitch, this.poseLo[1], this.poseHi[1]),
      outside(s.roll, this.poseLo[2], this.poseHi[2]),
    );
    return (worst * 180) / Math.PI;
  }
}

export function crossValidatedError(centres, targets, ridge = 1e-3, groups = null) {
  // Leave-one-group-out mean error in normalised screen units. Grouping by screen
  // target holds out all of a target's poses together, so no pose leaks.
  const n = centres.length;
  const labels = groups || centres.map((_, i) => i);
  const folds = [...new Set(labels)];
  if (folds.length < 3) throw new Error("need at least three groups for cross-validation");
  let total = 0, count = 0;
  for (const held of folds) {
    const trainSignals = [], trainTargets = [];
    for (let j = 0; j < n; j++) {
      if (labels[j] !== held) { trainSignals.push(centres[j]); trainTargets.push(targets[j]); }
    }
    const model = HeadAwareModel.fit(trainSignals, trainTargets, Math.max(ridge, 1e-6));
    for (let j = 0; j < n; j++) {
      if (labels[j] === held) {
        const [px, py] = model.map(centres[j]);
        total += Math.hypot(px - targets[j][0], py - targets[j][1]);
        count += 1;
      }
    }
  }
  return total / count;
}

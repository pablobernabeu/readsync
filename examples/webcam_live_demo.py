"""A self-contained, on-screen webcam eye-tracking test.

This is the script to run to see the webcam path working end to end on a laptop,
without an eye-tracker and without PsychoPy. It calibrates against on-screen
targets, then shows reading passages with a live gaze dot and the word currently
under gaze, and records everything to an encrypted, tamper-evident log that it
exports to CSV. It uses the readsync library for the gaze signals, the head-aware
calibration, the interest areas, the secure log and the export, and OpenCV only
for the camera and the window.

What makes it dependable in use:

* The iris feature is measured in the eye's own axes, so it does not break when
  you tilt your head.
* Calibration runs at a couple of head positions (set with --poses), so the model
  can learn how head yaw and pitch shift the mapping and compensate for them. The
  reading view turns the gaze amber and asks you to recalibrate once your head
  moves beyond the range you calibrated.
* Blinks are detected and hold the last gaze instead of throwing the dot.
* The calibration is checked by leave-one-out cross-validation, and a poor one is
  offered for redo before any reading starts.

Expectations. Even at its best, single-camera webcam gaze is coarse, on the
order of a degree or more of visual angle, and resolves a region of a line,
never a single letter. Sit with the screen well lit, keep your head reasonably
still, and recalibrate when prompted.

Requirements:

    pip install -e ".[webcam]"       # from the repository root: OpenCV and MediaPipe

The MediaPipe FaceLandmarker model file is needed once. If it is missing, the
script prints where to download it.

Controls:
    During calibration, look at each dot until it moves on.
    While reading, press SPACE for the next passage, C to recalibrate, ESC to stop.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from readsync import (
    EventLog,
    FixedWidthLayout,
    interest_areas,
    load_passages,
    locate,
    log_to_csv,
    new_data_key,
    pseudonymise,
)
from readsync.webcam import (
    FACE_LANDMARKER_URL,
    GazeSignals,
    HeadAwareGazeModel,
    MediaPipeIrisEstimator,
    aggregate_signals,
    cross_validated_error,
)

WINDOW = "readsync webcam eye-tracking demo"
WIDTH, HEIGHT = 1280, 720
BACKGROUND = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX

# A three by three grid of calibration targets, as fractions of the screen. The
# grid is shown once per head pose; calibrating at more than one pose gives the
# head-aware model the pose variation it needs to compensate, and pools enough
# points for cross-validation.
_AXIS = (0.1, 0.5, 0.9)
CALIB_TARGETS = [(x, y) for y in _AXIS for x in _AXIS]
SETTLE_SECONDS = 0.7
SAMPLES_PER_TARGET = 20

# A plausible iris feature sits near [0, 1] horizontally and near 0 vertically.
# Values well outside this come from a lost face, so they are dropped.
FX_RANGE, FY_RANGE = (-0.6, 1.6), (-1.6, 1.6)
# Eye-aspect ratio below this is treated as a blink; gaze is held, not updated.
BLINK_OPENNESS = 0.15
# Head rotation beyond this from the calibration pose is no longer trustworthy.
DRIFT_LIMIT_DEG = 12.0
# Accept a calibration whose leave-one-out error is within this fraction of screen.
MAX_CV_ERROR = 0.06
# Ridge penalty for the calibration fit.
RIDGE = 1e-3
# Exponential smoothing of the gaze point. Lower is steadier but lags more.
EMA_ALPHA = 0.35
# Give up on a camera that stops returning frames rather than spin forever.
MAX_DROPPED_FRAMES = 300

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "models" / "face_landmarker.task"
DEFAULT_STIMULI = HERE.parent / "stimuli" / "controlled_freq_en.json"


def _read_rgb(cap: cv2.VideoCapture) -> tuple[np.ndarray | None, np.ndarray | None]:
    ok, bgr = cap.read()
    if not ok:
        return None, None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _usable(signals: GazeSignals | None) -> bool:
    """True if a frame's signals are present, eyes open, and the feature sane."""
    if signals is None or signals.openness < BLINK_OPENNESS:
        return False
    return FX_RANGE[0] <= signals.fx <= FX_RANGE[1] and FY_RANGE[0] <= signals.fy <= FY_RANGE[1]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _thumbnail(canvas: np.ndarray, bgr: np.ndarray, found: bool) -> None:
    canvas[10:118, 10:202] = cv2.resize(cv2.flip(bgr, 1), (192, 108))
    colour = (0, 200, 0) if found else (0, 0, 200)
    cv2.rectangle(canvas, (10, 10), (202, 118), colour, 2)
    cv2.putText(canvas, "eyes found" if found else "no eyes", (16, 134), FONT, 0.5, colour, 1)


def _prompt(lines: list[str]) -> None:
    """Show an instruction screen and wait for SPACE. ESC quits."""
    while True:
        canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
        for k, line in enumerate(lines):
            cv2.putText(canvas, line, (60, 230 + k * 48), FONT, 0.85, (220, 220, 220), 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return
        if key == 27:
            raise KeyboardInterrupt


def _pose_instructions(index: int, total: int) -> list[str]:
    head = f"Calibration pose {index + 1} of {total}"
    if index == 0:
        return [head, "Sit naturally and look straight at the screen.",
                "Follow each dot with your eyes.", "Press SPACE to begin."]
    return [head, "Move to a slightly different head position",
            "(turn a little, or sit a little differently) and hold it.",
            "Keep following the dots. Press SPACE to begin."]


def run_calibration_pass(
    cap: cv2.VideoCapture, estimator: MediaPipeIrisEstimator
) -> tuple[list[list[GazeSignals]], list[tuple[float, float]]]:
    """Show each target once and collect usable gaze signals while it is fixated."""
    per_target: list[list[GazeSignals]] = []
    for tx, ty in CALIB_TARGETS:
        px, py = int(tx * WIDTH), int(ty * HEIGHT)
        samples: list[GazeSignals] = []
        start = time.perf_counter()
        dropped = 0
        while len(samples) < SAMPLES_PER_TARGET:
            bgr, rgb = _read_rgb(cap)
            if rgb is None:
                dropped += 1
                if dropped > MAX_DROPPED_FRAMES:
                    raise RuntimeError("camera stopped returning frames")
                if cv2.waitKey(1) == 27:
                    raise KeyboardInterrupt
                continue
            dropped = 0
            signals = estimator.signals(rgb)
            settling = (time.perf_counter() - start) < SETTLE_SECONDS
            if _usable(signals) and not settling:
                samples.append(signals)  # type: ignore[arg-type]

            canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
            cv2.circle(canvas, (px, py), 18, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(canvas, f"Look at the dot  ({len(samples)}/{SAMPLES_PER_TARGET})",
                        (40, HEIGHT - 40), FONT, 0.8, (200, 200, 200), 2)
            _thumbnail(canvas, bgr, _usable(signals))
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) == 27:  # Esc aborts
                raise KeyboardInterrupt
        per_target.append(samples)
    return per_target, list(CALIB_TARGETS)


def confirm_calibration(cv_error: float) -> bool:
    """Show the calibration accuracy and ask whether to use it. True to accept."""
    percent = cv_error * 100.0
    while True:
        canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
        ok = cv_error <= MAX_CV_ERROR
        colour = (0, 200, 0) if ok else (0, 165, 255)
        cv2.putText(canvas, f"Calibration error: {percent:.1f}% of the screen",
                    (60, 300), FONT, 1.0, colour, 2)
        verdict = "good" if ok else "poor; more light and a steadier head will help"
        cv2.putText(canvas, verdict, (60, 350), FONT, 0.8, colour, 2)
        cv2.putText(canvas, "SPACE use this    R redo    ESC quit", (60, 430),
                    FONT, 0.8, (200, 200, 200), 2)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            return True
        if key in (ord("r"), ord("R")):
            return False
        if key == 27:
            raise KeyboardInterrupt


def run_reading(
    cap: cv2.VideoCapture,
    estimator: MediaPipeIrisEstimator,
    model: HeadAwareGazeModel,
    passages: list,
    log: EventLog,
    start: int = 0,
    alpha: float = EMA_ALPHA,
) -> tuple[str, int]:
    """Show passages from ``start`` with a gaze dot. Returns (status, resume index)."""
    layout = FixedWidthLayout(char_width=20, line_height=54, max_chars_per_line=54, x0=90, y0=210)
    clock = time.perf_counter()
    smoothed: tuple[float, float] | None = None
    for idx in range(start, len(passages)):
        passage = passages[idx]
        areas = interest_areas(passage.words, layout)
        log.append({"type": "passage_onset", "t": time.perf_counter() - clock,
                    "passage": passage.id})
        current_word: int | None = None
        dropped = 0
        while True:
            _, rgb = _read_rgb(cap)
            if rgb is None:
                dropped += 1
                if dropped > MAX_DROPPED_FRAMES:
                    raise RuntimeError("camera stopped returning frames")
                if cv2.waitKey(1) == 27:
                    log.append({"type": "passage_offset", "t": time.perf_counter() - clock,
                                "passage": passage.id})
                    return "quit", idx
                continue
            dropped = 0
            t = time.perf_counter() - clock
            signals = estimator.signals(rgb)

            canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
            for area in areas:
                cv2.rectangle(canvas, (area.x1, area.y1), (area.x2, area.y2), (70, 70, 70), 1)
                cv2.putText(canvas, area.word.text, (area.x1 + 2, area.y2 - 16), FONT, 0.9,
                            (230, 230, 230), 2)

            note = ""
            if not _usable(signals):
                note = "blink" if signals is not None else "searching for your eyes"
            else:
                nx, ny = model.map(signals)  # type: ignore[arg-type]
                drift = model.pose_drift_degrees(signals)  # type: ignore[arg-type]
                if math.isfinite(nx) and math.isfinite(ny):
                    if smoothed is None:
                        smoothed = (nx, ny)
                    else:
                        smoothed = (alpha * nx + (1 - alpha) * smoothed[0],
                                    alpha * ny + (1 - alpha) * smoothed[1])
                    valid = drift <= DRIFT_LIMIT_DEG
                    gx = _clamp01(smoothed[0]) * (WIDTH - 1)
                    gy = _clamp01(smoothed[1]) * (HEIGHT - 1)
                    log.append({"type": "gaze", "t": t, "x": round(gx, 1), "y": round(gy, 1),
                                "valid": valid})
                    hit = locate(areas, gx, gy)
                    if hit is not None and valid:
                        cv2.rectangle(canvas, (hit.x1, hit.y1), (hit.x2, hit.y2), (0, 215, 255), 2)
                        if hit.word.index != current_word:
                            log.append({"type": "word_enter", "t": t, "passage": passage.id,
                                        "word": hit.word.index})
                            current_word = hit.word.index
                    cv2.circle(canvas, (int(gx), int(gy)), 9,
                               (0, 230, 0) if valid else (0, 165, 255), -1)
                    if not valid:
                        note = "you have moved; press C to recalibrate"
                else:
                    note = "tracking lost"

            if note:
                if smoothed is not None:  # hold the last gaze, dimmed, during a blink
                    lx = int(_clamp01(smoothed[0]) * (WIDTH - 1))
                    ly = int(_clamp01(smoothed[1]) * (HEIGHT - 1))
                    cv2.circle(canvas, (lx, ly), 7, (90, 90, 90), -1)
                cv2.putText(canvas, note, (40, 60), FONT, 0.7, (0, 165, 255), 2)
            cv2.putText(canvas, "SPACE next    C recalibrate    ESC quit", (40, HEIGHT - 30),
                        FONT, 0.7, (160, 160, 160), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                break
            if key == 27:
                log.append({"type": "passage_offset", "t": t, "passage": passage.id})
                return "quit", idx
            if key in (ord("c"), ord("C")):
                log.append({"type": "passage_offset", "t": t, "passage": passage.id})
                return "recalibrate", idx
        log.append({"type": "passage_offset", "t": time.perf_counter() - clock,
                    "passage": passage.id})
    return "done", len(passages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="FaceLandmarker model")
    parser.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI, help="stimulus JSON")
    parser.add_argument("--smoothing", type=float, default=EMA_ALPHA,
                        help="gaze smoothing in (0, 1]; lower is steadier but lags more")
    parser.add_argument("--poses", type=int, default=2,
                        help="head positions to calibrate at; 2 or more enables head-pose "
                             "compensation, 1 is a quick single-pose calibration")
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"Model file not found at {args.model}.")
        print("Download it once with, for example:")
        print(f"  curl -L -o {args.model} {FACE_LANDMARKER_URL}")
        return

    passages = load_passages(args.stimuli)
    key = new_data_key()
    participant = pseudonymise("webcam-demo", key=key)
    log = EventLog(f"data/{participant}.log", key=key)
    estimator = MediaPipeIrisEstimator(model_path=str(args.model))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)

    try:
        log.append({"type": "session_start", "participant": participant})
        index = 0
        while True:
            signals: list[GazeSignals] = []
            targets: list[tuple[float, float]] = []
            groups: list[int] = []
            for pose in range(max(1, args.poses)):
                _prompt(_pose_instructions(pose, max(1, args.poses)))
                per_target, grid = run_calibration_pass(cap, estimator)
                for i, samples in enumerate(per_target):
                    signals.append(aggregate_signals(samples))
                    targets.append(grid[i])
                    groups.append(i)
            model = HeadAwareGazeModel.fit(signals, targets, ridge=RIDGE)
            cv_error = cross_validated_error(signals, targets, ridge=RIDGE, groups=groups)
            print(f"calibration cross-validated error: {cv_error * 100:.1f}% of the screen")
            if not confirm_calibration(cv_error):
                continue
            status, index = run_reading(
                cap, estimator, model, passages, log, index, args.smoothing
            )
            if status != "recalibrate":
                break
    except KeyboardInterrupt:
        print("stopped early")
    finally:
        log.append({"type": "session_end", "participant": participant})
        cap.release()
        cv2.destroyAllWindows()

    out = log_to_csv(log, f"export/{participant}.csv")
    events = log.events()
    gaze = [e for e in events if e["type"] == "gaze"]
    valid = [e for e in gaze if e.get("valid")]
    words = [e for e in events if e["type"] == "word_enter"]
    print(f"recorded {len(events)} events ({len(gaze)} gaze, {len(valid)} trusted, "
          f"{len(words)} word entries)")
    print(f"exported to {out}")


if __name__ == "__main__":
    main()

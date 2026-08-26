"""Run a reading session on a webcam instead of the infrared tracker.

This shows the webcam counterpart slotting into the same session as the
research-grade device. Only the tracker changes: a :class:`WebcamTracker` driving
a camera and a gaze estimator replaces ``EyeLinkTracker``, and everything else,
the presentation, the offline encrypted log and the export, is unchanged.

Requirements:

    pip install -e ".[present,webcam]"   # from the repo root: PsychoPy, OpenCV, MediaPipe

Two caveats. Webcam gaze is far coarser than the infrared tracker and is
unsuitable for word-level reading measures. This path is for the methods strand
and for coarse uses, not for the main study. The estimator also needs a
per-participant calibration: the placeholder below is fitted from a few synthetic
points so the script runs as a wiring check, and a real study replaces it with a
calibration routine that records the eye while the reader looks at known targets.
With no camera attached the session still completes and simply records no gaze,
which makes it a safe smoke test of the wiring.
"""

from __future__ import annotations

from pathlib import Path

from readsync import (
    EventLog,
    FixedWidthLayout,
    PsychoPyPresenter,
    ReadingSession,
    WebcamTracker,
    load_passages,
    log_to_csv,
    new_data_key,
    pseudonymise,
)
from readsync.webcam import AffineCalibration, MediaPipeIrisEstimator

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / "models" / "face_landmarker.task"
STIMULI = HERE.parent / "stimuli" / "controlled_freq_en.json"


def _placeholder_calibration() -> AffineCalibration:
    # Stand-in only. A real study fits this from the reader's own eye while they
    # look at known screen positions. These synthetic pairs map a centred iris
    # feature to the middle of the screen so the wiring can be exercised.
    features = [(0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6), (0.5, 0.5)]
    targets = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5)]
    return AffineCalibration.fit(features, targets)


def main() -> None:
    key = new_data_key()
    participant = pseudonymise("demo-participant", key=key)
    passages = load_passages(STIMULI)

    size = (1280, 720)
    layout = FixedWidthLayout(char_width=16, line_height=40, max_chars_per_line=70, x0=80, y0=320)

    estimator = MediaPipeIrisEstimator(
        calibration=_placeholder_calibration(), model_path=str(MODEL)
    )
    tracker = WebcamTracker(estimator=estimator, screen_size=size, max_rate_hz=30.0)

    log = EventLog(f"data/{participant}.log", key=key)
    session = ReadingSession(
        participant=participant,
        tracker=tracker,
        presenter=PsychoPyPresenter(size=size, fullscreen=False),
        log=log,
        layout=layout,
    )

    result = session.run(passages)
    out = log_to_csv(log, f"export/{participant}.csv")
    print(f"recorded {result.n_events} events for {result.participant}")
    print(f"exported to {out}")


if __name__ == "__main__":
    main()

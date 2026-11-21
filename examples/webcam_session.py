"""Run a reading session with a webcam in place of the eye-tracker.

Only the tracker changes: a :class:`WebcamTracker` driving a camera and a gaze
estimator replaces ``EyeLinkTracker``, while the presentation, the encrypted log
and the export stay the same.

Requirements:

    pip install -e ".[present,webcam]"   # from the repository root: PsychoPy and MediaPipe

The estimator needs a calibration for each participant. The placeholder below is
fitted from a few synthetic points so that the script runs as a check of the
wiring. A study fits a ``HeadAwareGazeModel`` from the reader's own eyes, as
examples/webcam_live_demo.py does, and passes it as ``model``. With no camera
attached, the session still completes and records no gaze.
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
    # In a study, keep both keys apart from the data.
    data_key = new_data_key()
    participant = pseudonymise("demo-participant", key=new_data_key())
    passages = load_passages(STIMULI)

    size = (1280, 720)
    layout = FixedWidthLayout(char_width=16, line_height=40, max_chars_per_line=70, x0=80, y0=320)

    estimator = MediaPipeIrisEstimator(
        calibration=_placeholder_calibration(), model_path=str(MODEL)
    )
    tracker = WebcamTracker(estimator=estimator, screen_size=size, max_rate_hz=30.0)

    log = EventLog(f"data/{participant}-s1.log", key=data_key)
    session = ReadingSession(
        participant=participant,
        tracker=tracker,
        presenter=PsychoPyPresenter(size=size, fullscreen=False),
        log=log,
        layout=layout,
    )

    result = session.run(passages)
    out = log_to_csv(log, f"export/{participant}-s1.csv")
    print(f"recorded {result.n_events} events for {result.participant}")
    print(f"exported to {out}")


if __name__ == "__main__":
    main()

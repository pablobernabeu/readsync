"""Run a full reading session on a real screen.

This example shows the end-to-end path: pseudonymise a participant, open a
PsychoPy window, present passages with word-level interest areas, ask each
passage's comprehension question (answer with the Y and N keys), record to an
encrypted log offline, and export a CSV plus a per-session quality report.

Requirements:

    pip install -e ".[present]"      # from the repository root: PsychoPy and a display

Run it with the eye-tracker replaced by NullTracker, so it works on any machine
with a screen and no eye-tracker. In the lab, swap NullTracker for EyeLinkTracker
(or a Tobii backend) and the rest is unchanged. For the webcam methods strand,
swap in WebcamTracker instead; see examples/webcam_session.py. Press the space
bar to advance through each passage.
"""

from __future__ import annotations

from pathlib import Path

from readsync import (
    EventLog,
    FixedWidthLayout,
    NullTracker,
    PsychoPyPresenter,
    ReadingSession,
    load_stimulus_set,
    log_quality,
    log_to_csv,
    new_data_key,
    pseudonymise,
    quality_to_json,
)

STIMULI = Path(__file__).resolve().parent.parent / "stimuli" / "naturalistic_en.json"


def main() -> None:
    # In a real study, keep the key in the institution's secrets store, separate
    # from the data. Here it is generated for the demonstration only.
    key = new_data_key()
    participant = pseudonymise("demo-participant", key=key)
    items = load_stimulus_set(STIMULI).items

    size = (1280, 720)
    layout = FixedWidthLayout(char_width=16, line_height=40, max_chars_per_line=70, x0=80, y0=320)

    log = EventLog(f"data/{participant}.log", key=key)
    session = ReadingSession(
        participant=participant,
        # Swap NullTracker() for EyeLinkTracker() in the lab. The synthetic gaze
        # from NullTracker lets the screen path be exercised without a device.
        tracker=NullTracker(width=size[0], height=size[1], rate_hz=60),
        presenter=PsychoPyPresenter(size=size, fullscreen=False),
        log=log,
        layout=layout,
    )

    result = session.run_items(items)
    out = log_to_csv(log, f"export/{participant}.csv")
    report = log_quality(log, expected_rate_hz=60)
    report_path = quality_to_json(report, f"export/{participant}.quality.json")
    print(f"recorded {result.n_events} events for {result.participant}")
    print(f"comprehension: {report.n_correct}/{report.n_responses} correct")
    print(f"exported to {out}; quality report at {report_path}")


if __name__ == "__main__":
    main()

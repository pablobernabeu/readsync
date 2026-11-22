"""Run a full reading session on a real screen.

The example pseudonymises a participant, opens a PsychoPy window, presents
passages with word-level interest areas, asks each passage's comprehension
question (answer with the Y and N keys), records to an encrypted log and exports
the events, the interest areas and a quality report. Press the space bar to
advance through each passage.

Requirements:

    pip install -e ".[present]"      # from the repository root: PsychoPy and a display

NullTracker stands in for the eye-tracker, so the example runs on any machine
with a screen. With an EyeLink, the tracker is an EyeLinkTracker whose
screen_size matches the window. Open the presenter full screen at the display's
resolution and pass EyeLinkMarkerSink(tracker) as the marker sink, so that the
markers reach the data file. examples/webcam_session.py shows the webcam tracker in the same
place.
"""

from __future__ import annotations

from pathlib import Path

from readsync import (
    EventLog,
    FixedWidthLayout,
    NullTracker,
    PsychoPyPresenter,
    ReadingSession,
    interest_areas_to_csv,
    load_stimulus_set,
    log_quality,
    log_to_csv,
    new_data_key,
    pseudonymise,
    quality_to_json,
)

STIMULI = Path(__file__).resolve().parent.parent / "stimuli" / "naturalistic_en.json"


def main() -> None:
    # In a study, keep both keys in the institution's secrets store, apart from
    # the data. Here they are generated for the demonstration only.
    data_key = new_data_key()
    id_key = new_data_key()
    participant = pseudonymise("demo-participant", key=id_key)
    items = load_stimulus_set(STIMULI).items

    size = (1280, 720)
    layout = FixedWidthLayout(char_width=16, line_height=40, max_chars_per_line=70, x0=80, y0=320)

    log = EventLog(f"data/{participant}-s1.log", key=data_key)
    session = ReadingSession(
        participant=participant,
        # The synthetic gaze from NullTracker exercises the screen path without a
        # device.
        tracker=NullTracker(width=size[0], height=size[1], rate_hz=60),
        presenter=PsychoPyPresenter(size=size, fullscreen=False),
        log=log,
        layout=layout,
    )

    result = session.run_items(items)
    out = log_to_csv(log, f"export/{participant}-s1.csv")
    interest_areas_to_csv(items, "export/interest_areas.csv", layout)
    report = log_quality(log, expected_rate_hz=60)
    report_path = quality_to_json(report, f"export/{participant}-s1.quality.json")
    print(f"recorded {result.n_events} events for {result.participant}")
    print(f"comprehension: {report.n_correct}/{report.n_responses} correct")
    print(f"exported to {out}; quality report at {report_path}")


if __name__ == "__main__":
    main()

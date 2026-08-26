"""Tests for the per-session data-quality report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from readsync.quality import log_quality, quality_to_json, session_quality
from readsync.security import new_data_key
from readsync.session import HeadlessPresenter, ReadingSession
from readsync.storage import EventLog
from readsync.text import FixedWidthLayout, Passage
from readsync.trackers import NullTracker


def _recorded_log(tmp_path: Path) -> EventLog:
    log = EventLog(tmp_path / "s.log", key=new_data_key())
    session = ReadingSession(
        participant="p-anon-q",
        tracker=NullTracker(width=1920, height=1080, rate_hz=30),
        presenter=HeadlessPresenter(seconds_per_passage=1.0, rate_hz=30),
        log=log,
        layout=FixedWidthLayout(
            char_width=16, line_height=40, max_chars_per_line=80, x0=50, y0=525
        ),
    )
    session.run([Passage(id="p1", text="the cat sat"), Passage(id="p2", text="a dog ran")])
    return log


def test_report_summarises_a_recorded_session(tmp_path: Path) -> None:
    report = log_quality(_recorded_log(tmp_path), expected_rate_hz=30)
    assert report.participant == "p-anon-q"
    assert report.n_passages == 2
    assert report.n_samples > 0
    assert report.invalid_proportion == 0.0  # the NullTracker never blinks
    # the headless clock polls at the expected rate, so loss is near zero
    assert report.sample_loss is not None and report.sample_loss < 0.1
    assert len(report.passages) == 2
    assert all(p.duration > 0 for p in report.passages)


def test_report_counts_checks_and_responses() -> None:
    events = [
        {"type": "session_start", "participant": "p"},
        {"type": "calibration", "detail": "GOOD 0.3 deg"},
        {"type": "drift_check", "passage": "p1", "error": 0.5},
        {"type": "drift_check", "passage": "p2", "error": None},
        {"type": "passage_onset", "t": 0.0, "passage": "p1"},
        {"type": "gaze", "t": 0.0, "x": 1.0, "y": 1.0, "valid": True},
        {"type": "gaze", "t": 0.5, "x": 2.0, "y": 1.0, "valid": False},
        {"type": "gaze", "t": 1.0, "x": 3.0, "y": 1.0, "valid": True},
        {"type": "passage_offset", "t": 1.0, "passage": "p1"},
        {"type": "response", "t": 2.0, "onset": 1.5, "passage": "p1", "question": "q1",
         "kind": "literal", "region": None, "response": True, "correct": True},
        {"type": "response", "t": 3.0, "onset": 2.5, "passage": "p1", "question": "q2",
         "kind": "literal", "region": None, "response": False, "correct": False},
        {"type": "session_end", "participant": "p"},
    ]
    report = session_quality(events, expected_rate_hz=2)
    assert report.calibration == ["GOOD 0.3 deg"]
    assert report.drift_errors == [0.5, None]
    assert report.n_responses == 2 and report.n_correct == 1
    assert report.passages[0].n_samples == 3
    assert report.passages[0].invalid_proportion == round(1 / 3, 4)
    assert report.passages[0].largest_gap == 0.5
    assert report.observed_rate_hz == 2.0
    assert report.sample_loss == 0.0


def test_report_writes_json(tmp_path: Path) -> None:
    report = session_quality([], expected_rate_hz=None)
    out = quality_to_json(report, tmp_path / "sub" / "quality.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_passages"] == 0
    assert data["sample_loss"] is None


def test_unterminated_passage_is_reported_incomplete() -> None:
    events = [
        {"type": "session_start", "participant": "p"},
        {"type": "passage_onset", "t": 0.0, "passage": "p1"},
        {"type": "gaze", "t": 0.0, "x": 1.0, "y": 1.0, "valid": True},
        {"type": "gaze", "t": 0.5, "x": 2.0, "y": 1.0, "valid": True},
    ]
    report = session_quality(events)
    assert report.n_passages == 1
    assert report.n_samples == 2
    assert report.passages[0].complete is False
    assert report.passages[0].duration == 0.5


def test_single_sample_passage_cannot_zero_the_session_rate() -> None:
    events = [
        {"type": "passage_onset", "t": 0.0, "passage": "p1"},
        {"type": "gaze", "t": 0.0, "x": 1.0, "y": 1.0, "valid": True},
        {"type": "passage_offset", "t": 0.1, "passage": "p1"},
        {"type": "passage_onset", "t": 1.0, "passage": "p2"},
        {"type": "gaze", "t": 1.0, "x": 1.0, "y": 1.0, "valid": True},
        {"type": "gaze", "t": 1.5, "x": 2.0, "y": 1.0, "valid": True},
        {"type": "gaze", "t": 2.0, "x": 3.0, "y": 1.0, "valid": True},
        {"type": "passage_offset", "t": 2.0, "passage": "p2"},
    ]
    report = session_quality(events, expected_rate_hz=2)
    # The rate comes from the timed passage alone: two intervals over one second.
    assert report.observed_rate_hz == 2.0
    assert report.sample_loss == 0.0
    assert report.passages[0].n_samples == 1
    assert report.passages[0].duration == 0.0


def test_invalid_count_sums_raw_counts_not_rounded_proportions() -> None:
    # 10,001 samples with one invalid: the rounded per-passage proportion would
    # reconstruct zero invalid samples, so the report must carry raw counts.
    gaze = [
        {"type": "gaze", "t": i / 1000.0, "x": 1.0, "y": 1.0, "valid": i != 0}
        for i in range(10_001)
    ]
    events = [
        {"type": "passage_onset", "t": 0.0, "passage": "p1"},
        *gaze,
        {"type": "passage_offset", "t": 11.0, "passage": "p1"},
    ]
    report = session_quality(events)
    assert report.passages[0].n_invalid == 1
    assert report.invalid_proportion > 0.0


def test_expected_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        session_quality([], expected_rate_hz=0)

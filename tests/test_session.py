"""End-to-end test of a headless, offline reading session."""

from __future__ import annotations

from pathlib import Path

from readsync.markers import NullMarkerSink
from readsync.security import new_data_key
from readsync.session import HeadlessPresenter, ReadingSession
from readsync.stimuli import Question, StimulusItem
from readsync.storage import EventLog
from readsync.text import FixedWidthLayout, Passage, Region
from readsync.trackers import GazeSample, NullTracker


def test_headless_session_records_an_auditable_log(tmp_path: Path) -> None:
    key = new_data_key()
    log = EventLog(tmp_path / "s.log", key=key)
    markers = NullMarkerSink()
    # Place the text where the NullTracker's synthetic gaze sweeps (screen mid),
    # so the run produces region-entry events as a real reading session would.
    session = ReadingSession(
        participant="p-anon-01",
        tracker=NullTracker(width=1920, height=1080, rate_hz=30),
        presenter=HeadlessPresenter(seconds_per_passage=0.5, rate_hz=30),
        log=log,
        marker_sink=markers,
        layout=FixedWidthLayout(
            char_width=16, line_height=40, max_chars_per_line=80, x0=50, y0=525
        ),
    )
    passages = [Passage(id="p1", text="the cat sat on the mat"), Passage(id="p2", text="a dog ran")]
    result = session.run(passages)

    assert result.n_passages == 2
    events = log.events()  # also verifies the integrity chain
    types = {e["type"] for e in events}
    assert {"session_start", "passage_onset", "gaze", "passage_offset", "session_end"} <= types
    assert any(m.label == "passage_onset" for m in markers.markers)
    # the synthetic gaze should land on at least one word
    assert any(e["type"] == "word_enter" for e in events)


def test_session_can_run_without_the_offline_guard(tmp_path: Path) -> None:
    key = new_data_key()
    log = EventLog(tmp_path / "s.log", key=key)
    session = ReadingSession(
        participant="p-anon-02",
        tracker=NullTracker(),
        presenter=HeadlessPresenter(seconds_per_passage=0.2),
        log=log,
        offline=False,
    )
    result = session.run([Passage(id="p1", text="hello world")])
    assert result.n_events == len(log)


class _CheckedTracker(NullTracker):
    """A NullTracker that also offers the optional quality-check methods."""

    def __init__(self, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.last_calibration_message = "GOOD 0.31 deg"
        self.drift_calls = 0

    def drift_correct(self) -> float | None:
        self.drift_calls += 1
        return 0.4


def _annotated_item(question_position: str = "after") -> StimulusItem:
    passage = Passage(id="p1", text="the cat sat on the mat")
    return StimulusItem(
        passage=passage,
        regions=[
            Region(id="target", start=1, end=3, layer="decoding"),
            Region(id="control", start=4, end=6, layer="decoding", role="comparison"),
        ],
        questions=[
            Question(id="q1", text="Did the cat sit?", answer=True, region="target"),
            Question(id="q2", text="Was there a dog?", answer=False, kind="inferential"),
        ],
        question_position=question_position,
    )


def _annotated_session(
    tmp_path: Path, *, question_position: str = "after"
) -> tuple[ReadingSession, EventLog, _CheckedTracker]:
    log = EventLog(tmp_path / "s.log", key=new_data_key())
    tracker = _CheckedTracker(width=1920, height=1080, rate_hz=30)
    session = ReadingSession(
        participant="p-anon-03",
        tracker=tracker,
        presenter=HeadlessPresenter(
            seconds_per_passage=2.5,
            rate_hz=30,
            scripted_responses={"q1": True, "q2": True},
        ),
        log=log,
        layout=FixedWidthLayout(
            char_width=16, line_height=40, max_chars_per_line=80, x0=50, y0=525
        ),
    )
    return session, log, tracker


def test_run_items_emits_region_events_with_their_layer(tmp_path: Path) -> None:
    session, log, _ = _annotated_session(tmp_path)
    session.run_items([_annotated_item()])
    events = log.events()
    enters = [e for e in events if e["type"] == "region_enter"]
    exits = [e for e in events if e["type"] == "region_exit"]
    assert {e["region"] for e in enters} == {"target", "control"}
    assert all(e["layer"] == "decoding" for e in enters)
    # every entered region is exited exactly once, at the latest when the
    # passage ends, and blinks or between-word gaze never churn extra pairs
    assert len(exits) == len(enters)


def test_run_items_scores_and_logs_the_responses(tmp_path: Path) -> None:
    session, log, _ = _annotated_session(tmp_path)
    session.run_items([_annotated_item()])
    responses = {e["question"]: e for e in log.events() if e["type"] == "response"}
    assert responses["q1"]["correct"] is True  # scripted yes, answer yes
    assert responses["q2"]["correct"] is False  # scripted yes, answer no
    assert responses["q1"]["region"] == "target"
    assert responses["q2"]["kind"] == "inferential"
    assert responses["q1"]["t"] > responses["q1"]["onset"]


def test_information_seeking_regime_shows_the_questions_first(tmp_path: Path) -> None:
    session, log, _ = _annotated_session(tmp_path, question_position="before")
    session.run_items([_annotated_item(question_position="before")])
    events = log.events()
    types = [e["type"] for e in events]
    assert types.index("prompt") < types.index("passage_onset")
    assert sum(1 for t in types if t == "prompt") == 2


def test_tracker_checks_are_logged_for_the_quality_report(tmp_path: Path) -> None:
    session, log, tracker = _annotated_session(tmp_path)
    session.run_items([_annotated_item()])
    events = log.events()
    assert any(
        e["type"] == "calibration" and "GOOD" in e["detail"] for e in events
    )
    drift = [e for e in events if e["type"] == "drift_check"]
    assert len(drift) == 1 and drift[0]["error"] == 0.4
    assert tracker.drift_calls == 1


class _ScriptedTracker:
    """Replays a fixed sequence of gaze samples, one per poll."""

    def __init__(self, script: list[GazeSample | None]) -> None:
        self._script = list(script)

    def connect(self) -> None:
        return None

    def calibrate(self) -> None:
        return None

    def start_recording(self) -> None:
        return None

    def stop_recording(self) -> None:
        return None

    def poll(self, t: float) -> GazeSample | None:
        if not self._script:
            return None
        sample = self._script.pop(0)
        if sample is None:
            return None
        return GazeSample(t=t, x=sample.x, y=sample.y, valid=sample.valid)

    def close(self) -> None:
        return None


def _at_word(x: float, *, valid: bool = True) -> GazeSample:
    return GazeSample(t=0.0, x=x, y=540.0, valid=valid)


def test_region_markers_fire_on_genuine_transitions_only(tmp_path: Path) -> None:
    # Layout: "the cat sat on the mat" at x0=50, char width 16. 'cat' (word 1,
    # region target) spans x 114-162; 'the' (word 4, region control) 290-338;
    # x=170 lands in the gap after 'cat'.
    script = [
        _at_word(120.0),                     # fixate 'cat': word_enter, enter target
        _at_word(120.0, valid=False),        # blink: everything holds
        _at_word(170.0),                     # gap between words: everything holds
        _at_word(125.0),                     # same word again: no new events
        _at_word(300.0),                     # 'the' (word 4): exit target, enter control
        _at_word(300.0),                     # steady: no new events
    ]
    log = EventLog(tmp_path / "s.log", key=new_data_key())
    session = ReadingSession(
        participant="p-anon-04",
        tracker=_ScriptedTracker(script),
        presenter=HeadlessPresenter(seconds_per_passage=0.2, rate_hz=30),
        log=log,
        layout=FixedWidthLayout(
            char_width=16, line_height=40, max_chars_per_line=80, x0=50, y0=525
        ),
    )
    session.run_items([_annotated_item()])
    events = log.events()
    sequence = [
        (e["type"], e.get("word", e.get("region")))
        for e in events
        if e["type"] in ("word_enter", "region_enter", "region_exit")
    ]
    assert sequence == [
        ("word_enter", 1),
        ("region_enter", "target"),
        ("word_enter", 4),
        ("region_exit", "target"),
        ("region_enter", "control"),
        ("region_exit", "control"),  # emitted at passage end
    ]


def test_gaze_events_carry_the_session_clock_and_the_tracker_clock(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s.log", key=new_data_key())
    session = ReadingSession(
        participant="p-anon-05",
        tracker=NullTracker(rate_hz=30),
        presenter=HeadlessPresenter(seconds_per_passage=0.1, rate_hz=30),
        log=log,
    )
    session.run([Passage(id="p1", text="hello world")])
    gaze = [e for e in log.events() if e["type"] == "gaze"]
    assert gaze and all("tracker_t" in e and "t" in e for e in gaze)


def test_questions_are_skipped_when_the_presenter_cannot_ask(tmp_path: Path) -> None:
    class _MutePresenter:
        """Implements only the core protocol, with no ask or show_prompt."""

        def start_passage(self, passage: object, areas: object) -> None:
            self._i = 0

        def tick(self) -> tuple[float, bool]:
            self._i += 1
            return self._i / 30.0, self._i >= 3

        def end_passage(self) -> None:
            return None

        def close(self) -> None:
            return None

    log = EventLog(tmp_path / "s.log", key=new_data_key())
    session = ReadingSession(
        participant="p-anon-06",
        tracker=NullTracker(),
        presenter=_MutePresenter(),
        log=log,
    )
    result = session.run_items([_annotated_item(question_position="before")])
    types = {e["type"] for e in log.events()}
    assert result.n_passages == 1
    assert "response" not in types
    assert "prompt" not in types

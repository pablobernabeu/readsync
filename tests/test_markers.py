"""Tests for marker serialisation into the hardware streams."""

from __future__ import annotations

from readsync.markers import EYELINK_MESSAGE_LIMIT, EyeLinkMarkerSink, Marker


class _RecordingTracker:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_message(self, text: str) -> None:
        self.messages.append(text)


def test_marker_message_carries_sorted_metadata() -> None:
    marker = Marker("region_enter", 1.5, {"region": "r1", "passage": "p1", "layer": "decoding"})
    assert marker.message() == "region_enter layer=decoding passage=p1 region=r1"


def test_marker_message_without_metadata_is_the_bare_label() -> None:
    assert Marker("session_start", 0.0).message() == "session_start"


def test_eyelink_sink_sends_the_full_message() -> None:
    tracker = _RecordingTracker()
    sink = EyeLinkMarkerSink(tracker)
    sink.send(Marker("word_enter", 2.0, {"passage": "p1", "word": 12}))
    assert tracker.messages == ["word_enter passage=p1 word=12"]
    assert sink.markers[0].label == "word_enter"


def test_marker_message_sanitises_whitespace_and_equals() -> None:
    marker = Marker("region_enter", 1.0, {"region": "critical spillover=2", "passage": "p 1"})
    assert marker.message() == "region_enter passage=p_1 region=critical_spillover_2"


def test_eyelink_sink_caps_the_message_length() -> None:
    tracker = _RecordingTracker()
    sink = EyeLinkMarkerSink(tracker)
    sink.send(Marker("word_enter", 0.0, {"passage": "x" * 300}))
    assert len(tracker.messages[0]) == EYELINK_MESSAGE_LIMIT
    # The in-memory copy keeps the full marker for auditing.
    assert len(sink.markers[0].meta["passage"]) == 300

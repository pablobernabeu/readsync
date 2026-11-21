"""Tests for marker serialisation into the hardware streams."""

from __future__ import annotations

import pytest

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


def test_multi_sink_sends_each_marker_to_every_sink_in_order() -> None:
    from readsync.markers import MultiMarkerSink, NullMarkerSink

    calls: list[str] = []

    class _Named(NullMarkerSink):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def send(self, marker: Marker) -> None:
            calls.append(self.name)
            super().send(marker)

    first, second = _Named("eyelink"), _Named("eeg")
    MultiMarkerSink(first, second).send(Marker("passage_onset", 0.0, {"passage": "p1"}))
    assert calls == ["eyelink", "eeg"]
    assert first.markers == second.markers
    with pytest.raises(ValueError):
        MultiMarkerSink()


def test_lsl_sink_opens_its_outlet_before_the_first_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from readsync.markers import LSLMarkerSink

    created: list[object] = []

    class _Outlet:
        def __init__(self, info: object) -> None:
            self.info = info
            self.samples: list[list[str]] = []
            created.append(self)

        def wait_for_consumers(self, timeout: float) -> bool:
            return timeout > 0

        def push_sample(self, sample: list[str]) -> None:
            self.samples.append(sample)

    fake = types.ModuleType("pylsl")
    fake.StreamInfo = lambda *args: args  # type: ignore[attr-defined]
    fake.StreamOutlet = _Outlet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pylsl", fake)

    sink = LSLMarkerSink(stream_name="test-markers")
    assert len(created) == 1  # the stream exists before any marker is sent
    assert sink.wait_for_recorder(1.0)
    sink.send(Marker("passage_onset", 0.0, {"passage": "p1"}))
    assert created[0].samples == [["passage_onset passage=p1"]]  # type: ignore[attr-defined]

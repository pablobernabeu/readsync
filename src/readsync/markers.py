"""Event markers for synchronising stimulus, eye-tracker and EEG.

To align a gaze or EEG record with what was on screen, the same event has to be
time-stamped in every stream. A marker is a short labelled message sent at a
known moment, for example when a passage appears or when the gaze enters a
region. ``MarkerSink`` is the interface. The sinks send markers to the
eye-tracker's data file, to a Lab Streaming Layer stream for the EEG recording,
to several sinks at once or, in development, to an in-memory list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EyeLinkMarkerSink",
    "LSLMarkerSink",
    "Marker",
    "MarkerSink",
    "MultiMarkerSink",
    "NullMarkerSink",
]

_UNSAFE = re.compile(r"[\s=]+")

# EyeLink data-file messages have a maximum length. pylink's documentation gives
# 130 characters and PsychoPy's SR Research plugin 128, so markers are capped at
# the lower figure, which makes any truncation explicit and the same everywhere.
EYELINK_MESSAGE_LIMIT = 128

# Marker fields that identify the event, written before the others.
_IDENTIFIERS = ("passage", "word", "region", "question")


def _token(value: Any) -> str:
    """One whitespace- and ``=``-free token for the ``key=value`` message format."""
    return _UNSAFE.sub("_", str(value))


@dataclass(frozen=True)
class Marker:
    """A labelled event. ``t`` is seconds from session start; ``meta`` carries
    structured detail such as the passage id, the word index, or the region id
    and its layer."""

    label: str
    t: float
    meta: dict[str, Any] = field(default_factory=dict)

    def message(self) -> str:
        """The marker as one line for a hardware stream.

        The label is followed by the metadata as ``key=value`` pairs, the
        identifiers first (passage, word, region, question) and the rest in
        alphabetical order, for example
        ``region_enter passage=p1 region=r1 layer=decoding``. Putting the
        identifiers first means that a message cut to a length limit loses the
        least important fields. Whitespace and ``=`` inside the label, keys or
        values are replaced with ``_``, so a study-supplied id cannot make the
        line unparseable, and the stimulus loader refuses such ids at load.
        """
        order = {key: rank for rank, key in enumerate(_IDENTIFIERS)}
        fields = sorted(self.meta.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))
        detail = " ".join(f"{_token(k)}={_token(v)}" for k, v in fields)
        return f"{_token(self.label)} {detail}".strip()


@runtime_checkable
class MarkerSink(Protocol):
    def send(self, marker: Marker) -> None: ...


class NullMarkerSink:
    """Records markers in memory. Used for development and tests, and as the
    fallback when no hardware marker channel is configured."""

    def __init__(self) -> None:
        self.markers: list[Marker] = []

    def send(self, marker: Marker) -> None:
        self.markers.append(marker)


class LSLMarkerSink:
    """Send markers on a Lab Streaming Layer outlet for EEG co-registration.

    Lab Streaming Layer carries the marker stream over the local network to a
    recorder that saves it alongside the EEG. The outlet is created when the
    sink is constructed, so the stream can be selected in the recorder before
    the session starts, and :meth:`wait_for_recorder` confirms that one has
    connected. Markers pushed before a recorder connects are not delivered.
    The sink keeps working under ``NetworkGuard`` because ``pylsl`` binds the
    native ``liblsl`` library, whose sockets bypass the Python socket layer
    that the guard patches. ``pylsl`` is imported only when a sink is created,
    so this module imports without it.
    """

    def __init__(self, *, stream_name: str = "readsync-markers") -> None:
        self.stream_name = stream_name
        try:
            from pylsl import StreamInfo, StreamOutlet
        except ImportError as exc:
            raise RuntimeError(
                "pylsl is not installed. Install the 'lsl' extra to send EEG "
                "markers, or use NullMarkerSink."
            ) from exc
        info = StreamInfo(stream_name, "Markers", 1, 0, "string", stream_name)
        self._outlet: Any = StreamOutlet(info)

    def wait_for_recorder(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for a recorder to connect.

        Returns whether one is connected. Call it before the session starts, so
        the first markers are not lost.
        """
        return bool(self._outlet.wait_for_consumers(timeout))

    def send(self, marker: Marker) -> None:
        self._outlet.push_sample([marker.message()])


class EyeLinkMarkerSink:
    """Route markers into the EyeLink data file for eye-record synchronisation.

    The EyeLink stamps a message on the same clock as its gaze samples, so sending
    each marker as a message aligns on-screen events with the eye movements in
    the data file. The sink wraps anything with a ``send_message`` method, so it
    needs no hardware to import or test. Markers are also kept in memory for
    cross-checking against the data file. Messages are capped at
    :data:`EYELINK_MESSAGE_LIMIT` characters, and the in-memory copy keeps the
    full marker.
    """

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker
        self.markers: list[Marker] = []

    def send(self, marker: Marker) -> None:
        self.markers.append(marker)
        self._tracker.send_message(marker.message()[:EYELINK_MESSAGE_LIMIT])


class MultiMarkerSink:
    """Send each marker to several sinks, in the order given.

    The session takes one sink, so this is how the same marker reaches the
    eye-tracker's data file and the EEG stream, for example
    ``MultiMarkerSink(EyeLinkMarkerSink(tracker), LSLMarkerSink())``. The sinks
    are called one after another, so each later sink receives the marker
    slightly later. Put the stream whose timing matters most first.
    """

    def __init__(self, *sinks: MarkerSink) -> None:
        if not sinks:
            raise ValueError("MultiMarkerSink needs at least one sink")
        self.sinks: tuple[MarkerSink, ...] = sinks

    def send(self, marker: Marker) -> None:
        for sink in self.sinks:
            sink.send(marker)

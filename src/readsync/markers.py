"""Event markers for synchronising stimulus, eye-tracker and EEG.

To align a neural or gaze record with what was on screen, the same event has to
be timestamped in every stream. A marker is a short labelled pulse sent at a
known moment, for example when a passage appears or when a region is first
fixated. ``MarkerSink`` is the interface; concrete sinks send the marker to the
eye-tracker, to an EEG amplifier over Lab Streaming Layer, or, in development, to
an in-memory list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Marker", "MarkerSink", "NullMarkerSink", "LSLMarkerSink", "EyeLinkMarkerSink"]

_UNSAFE = re.compile(r"[\s=]+")

# The EyeLink Programmer's Guide advises keeping data-file messages short; long
# messages are truncated by the Host PC. Capping here makes the truncation
# explicit and deterministic instead of silent and device-dependent.
EYELINK_MESSAGE_LIMIT = 130


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

        The label is followed by the metadata as sorted ``key=value`` pairs, so
        the EyeLink data file and the EEG stream carry the same detail as the
        session log and alignment needs no cross-referencing, for example
        ``region_enter layer=decoding passage=p1 region=r1``. Whitespace and
        ``=`` inside the label, keys or values are replaced with ``_``, so a
        study-supplied id can never make the line unparseable; the stimulus
        loader additionally refuses such ids at load time.
        """
        detail = " ".join(f"{_token(k)}={_token(v)}" for k, v in sorted(self.meta.items()))
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

    LSL is local inter-process transport and keeps working during an offline
    session because ``pylsl`` binds the native ``liblsl`` library, whose
    sockets sit outside the Python socket layer that ``NetworkGuard`` patches.
    ``pylsl`` is imported lazily so this module imports without it. This is a
    stub for the lab build.
    """

    def __init__(self, *, stream_name: str = "readsync-markers") -> None:
        self.stream_name = stream_name
        self._outlet: object | None = None

    def _ensure_outlet(self) -> object:  # pragma: no cover - hardware path
        if self._outlet is None:
            try:
                from pylsl import StreamInfo, StreamOutlet
            except ImportError as exc:
                raise RuntimeError(
                    "pylsl is not installed. Install the 'lsl' extra to send EEG "
                    "markers, or use NullMarkerSink."
                ) from exc
            info = StreamInfo(self.stream_name, "Markers", 1, 0, "string", self.stream_name)
            self._outlet = StreamOutlet(info)
        return self._outlet

    def send(self, marker: Marker) -> None:  # pragma: no cover - hardware path
        outlet = self._ensure_outlet()
        outlet.push_sample([marker.message()])  # type: ignore[attr-defined]


class EyeLinkMarkerSink:
    """Route markers into the EyeLink data file for eye-record synchronisation.

    The EyeLink stamps a message in the same clock as its gaze samples, so sending
    each marker as a message is how on-screen events are aligned with the eye
    movements during analysis. It wraps anything with a ``send_message`` method,
    so it is decoupled from the tracker class and needs no hardware to import.
    Markers are also kept in memory for inspection and for cross-checking against
    the data file. Messages are capped at :data:`EYELINK_MESSAGE_LIMIT`
    characters, because the Host PC truncates long messages; the in-memory copy
    keeps the full marker.
    """

    def __init__(self, tracker: Any) -> None:
        self._tracker = tracker
        self.markers: list[Marker] = []

    def send(self, marker: Marker) -> None:
        self.markers.append(marker)
        self._tracker.send_message(marker.message()[:EYELINK_MESSAGE_LIMIT])

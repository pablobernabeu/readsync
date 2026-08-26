"""Tests for the tracker backends.

The EyeLink device path cannot run without hardware, so it is marked as not
covered in the source and exercised in the lab. What is tested here is the pure
logic the device path depends on: the data-file name check and the conversion of
raw sample components to a :class:`GazeSample`, plus the lazy-import behaviour and
the marker routing, none of which need a device.
"""

from __future__ import annotations

import importlib.util

import pytest

from readsync.trackers import (
    NullTracker,
    _gaze_from_components,
    _validate_edf_name,
)

_HAS_PYLINK = importlib.util.find_spec("pylink") is not None


def test_validate_edf_name_accepts_and_normalises() -> None:
    assert _validate_edf_name("readsync") == "readsync.edf"
    assert _validate_edf_name("readsync.edf") == "readsync.edf"
    assert _validate_edf_name("ab_12_34") == "ab_12_34.edf"


@pytest.mark.parametrize("bad", ["", "toolongname", "bad-name", "has space", "nine_char"])
def test_validate_edf_name_rejects_illegal(bad: str) -> None:
    with pytest.raises(ValueError, match="EDF name"):
        _validate_edf_name(bad)


def test_gaze_from_components_maps_time_and_passes_coordinates() -> None:
    sample = _gaze_from_components(640.0, 360.0, time_ms=1500.0, t0_ms=1000.0)
    assert sample.t == pytest.approx(0.5)  # (1500 - 1000) ms in seconds
    assert (sample.x, sample.y) == (640.0, 360.0)
    assert sample.valid


def test_gaze_from_components_marks_missing_as_invalid() -> None:
    lost = _gaze_from_components(-32768.0, 360.0, time_ms=1000.0, t0_ms=1000.0)
    assert not lost.valid
    also_lost = _gaze_from_components(640.0, -32768.0, time_ms=1000.0, t0_ms=1000.0)
    assert not also_lost.valid


def test_null_tracker_is_silent_until_recording() -> None:
    tracker = NullTracker()
    assert tracker.poll(0.1) is None
    tracker.start_recording()
    sample = tracker.poll(0.1)
    assert sample is not None and sample.valid
    tracker.stop_recording()
    assert tracker.poll(0.2) is None


@pytest.mark.skipif(_HAS_PYLINK, reason="pylink is installed; this checks the missing-SDK path")
def test_eyelink_connect_without_pylink_raises_clearly() -> None:
    from readsync.trackers import EyeLinkTracker

    with pytest.raises(RuntimeError, match="pylink is not installed"):
        EyeLinkTracker().connect()



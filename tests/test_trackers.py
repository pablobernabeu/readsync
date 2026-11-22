"""Tests for the tracker backends.

The EyeLink device path cannot run without hardware, so it is marked as not
covered in the source and exercised in the lab. What is tested here is the pure
logic the device path depends on: the data-file name check and the conversion of
raw sample components to a :class:`GazeSample`, plus the lazy-import behaviour and
the marker routing, none of which need a device.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


@pytest.mark.parametrize(
    "bad", ["", "toolongname", "bad-name", "has space", "nine_char", "café", "_s01"]
)
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


def test_eyelink_connect_without_pylink_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    from readsync.trackers import EyeLinkTracker

    monkeypatch.setitem(sys.modules, "pylink", None)  # makes `import pylink` fail
    with pytest.raises(RuntimeError, match="pylink is not installed"):
        EyeLinkTracker().connect()




class _FakeLink:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def sendMessage(self, text: str) -> None:
        self.messages.append(text)


def test_eyelink_trial_messages_and_interest_area_file(tmp_path: Path) -> None:
    from readsync.text import FixedWidthLayout, interest_areas, tokenise
    from readsync.trackers import EyeLinkTracker

    tracker = EyeLinkTracker(edf_name="s01", receive_to=str(tmp_path / "s01.edf"))
    tracker._link = _FakeLink()
    areas = interest_areas(tokenise("a cat"), FixedWidthLayout())
    tracker.begin_trial("p01h", areas)
    tracker.end_trial({"passage": "p01h", "condition": "high"})
    assert tracker._link.messages == [
        "TRIALID p01h",
        "!V IAREA FILE s01_001.ias",
        "!V TRIAL_VAR passage p01h",
        "!V TRIAL_VAR condition high",
        "TRIAL_RESULT 0",
    ]
    lines = (tmp_path / "s01_001.ias").read_text(encoding="utf-8").splitlines()
    assert lines[1].split("\t")[:2] == ["RECTANGLE", "1"]
    assert len(lines) == 3


def test_eyelink_trial_messages_without_interest_area_files(tmp_path: Path) -> None:
    from readsync.trackers import EyeLinkTracker

    tracker = EyeLinkTracker(receive_to=str(tmp_path / "s.edf"), interest_area_files=False)
    tracker._link = _FakeLink()
    tracker.begin_trial("p1", [])
    assert tracker._link.messages == ["TRIALID p1"]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not _HAS_PYLINK, reason="needs SR Research's pylink (sr-research-pylink)")
def test_eyelink_call_sequence_runs_in_pylink_dummy_mode(tmp_path: Path) -> None:
    from readsync.text import FixedWidthLayout, interest_areas, tokenise
    from readsync.trackers import EyeLinkTracker

    tracker = EyeLinkTracker(dummy=True, edf_name="dummy", receive_to=str(tmp_path / "d.edf"))
    tracker.connect()
    tracker.begin_trial("p1", interest_areas(tokenise("a cat"), FixedWidthLayout()))
    tracker.start_recording()
    assert tracker.poll(0.0) is None  # dummy mode produces no samples
    tracker.stop_recording()
    tracker.end_trial({"passage": "p1"})
    tracker.close()
    assert (tmp_path / "d_001.ias").exists()


def test_eyelink_builds_calibration_graphics_from_the_connected_link() -> None:
    from readsync.trackers import EyeLinkTracker

    class _Pylink:
        def __init__(self) -> None:
            self.opened: list[object] = []

        def openGraphicsEx(self, graphics: object) -> None:
            self.opened.append(graphics)

    class _Link(_FakeLink):
        def doTrackerSetup(self) -> None:
            return None

        def getCalibrationMessage(self) -> str:
            return "calibration ok"

    built: list[object] = []

    def factory(link: object) -> object:
        built.append(link)
        return ("graphics for", link)

    tracker = EyeLinkTracker(graphics_factory=factory)
    tracker._pylink = _Pylink()
    tracker._link = _Link()
    tracker.calibrate()
    tracker.calibrate()  # graphics are opened once per connection
    assert built == [tracker._link]
    assert tracker._pylink.opened == [("graphics for", tracker._link)]
    assert tracker.last_calibration_message == "calibration ok"


class _Eye:
    def __init__(self, x: float, y: float) -> None:
        self._xy = (x, y)

    def getGaze(self) -> tuple[float, float]:
        return self._xy


class _Sample:
    def __init__(self, left: tuple[float, float] | None, right: tuple[float, float] | None) -> None:
        self._left, self._right = left, right

    def isLeftSample(self) -> bool:
        return self._left is not None

    def isRightSample(self) -> bool:
        return self._right is not None

    def getLeftEye(self) -> _Eye:
        assert self._left is not None
        return _Eye(*self._left)

    def getRightEye(self) -> _Eye:
        assert self._right is not None
        return _Eye(*self._right)

    def getTime(self) -> float:
        return 5000.0


@pytest.mark.parametrize(
    ("eye", "left", "right", "expected"),
    [
        (None, (10.0, 20.0), (30.0, 40.0), (30.0, 40.0, True)),
        (None, (10.0, 20.0), None, (10.0, 20.0, True)),
        ("left", (10.0, 20.0), (30.0, 40.0), (10.0, 20.0, True)),
        ("left", None, (30.0, 40.0), (-32768.0, -32768.0, False)),
    ],
)
def test_eyelink_poll_uses_the_chosen_eye(
    eye: str | None,
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    expected: tuple[float, float, bool],
) -> None:
    from readsync.trackers import EyeLinkTracker

    class _Link:
        def getNewestSample(self) -> _Sample:
            return _Sample(left, right)

    tracker = EyeLinkTracker(eye=eye)
    tracker._link = _Link()
    sample = tracker.poll(0.0)
    assert sample is not None
    assert (sample.x, sample.y, sample.valid) == expected
    assert sample.t == 5.0  # the Host PC time of the sample, in seconds


def test_eyelink_refuses_an_unknown_eye() -> None:
    from readsync.trackers import EyeLinkTracker

    with pytest.raises(ValueError, match="eye must be"):
        EyeLinkTracker(eye="both")

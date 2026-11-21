"""Eye-tracker backends behind a small common interface.

The session is written against a ``Tracker`` protocol, so the experiment logic
does not depend on a particular device and runs with no hardware at all.

``NullTracker`` produces synthetic gaze for development and tests and must never
be used to collect data. ``EyeLinkTracker`` drives an SR Research EyeLink through
pylink, following SR Research's documented call sequence. Its connection, trial,
recording and closing calls run in pylink's dummy mode, while calibration, the
drift check and recording still have to be validated with a tracker. pylink is
imported only when the tracker connects, so importing this module never requires
it. Other backends, such as the webcam tracker in ``readsync.webcam``, implement
the same protocol.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .dataviewer import ias_filename, trial_end_messages, trial_start_messages, write_ias
from .text import InterestArea

__all__ = ["GazeSample", "Tracker", "NullTracker", "EyeLinkTracker"]

# The EyeLink reports a coordinate equal to this value when an eye is lost, for
# example during a blink. pylink exposes it as ``pylink.MISSING_DATA``. It is
# repeated here so the validity logic can be tested without the SDK.
_MISSING_DATA = -32768.0


@dataclass(frozen=True)
class GazeSample:
    """One gaze estimate. ``t`` is the time in seconds on the tracker's own clock,
    or the session time for a tracker without one; ``x``/``y`` are pixels;
    ``valid`` is false during blinks or track loss."""

    t: float
    x: float
    y: float
    valid: bool = True


@runtime_checkable
class Tracker(Protocol):
    """The minimal interface the session needs from an eye-tracker."""

    def connect(self) -> None: ...
    def calibrate(self) -> None: ...
    def start_recording(self) -> None: ...
    def stop_recording(self) -> None: ...
    def poll(self, t: float) -> GazeSample | None: ...
    def close(self) -> None: ...


class NullTracker:
    """A hardware-free tracker that returns deterministic synthetic gaze.

    It traces a slow left-to-right sweep so that sessions, logging and export can
    be exercised end to end without a device. It is for development and testing
    only and must never be used to collect research data.
    """

    def __init__(self, *, width: int = 1920, height: int = 1080, rate_hz: int = 60) -> None:
        self.width = width
        self.height = height
        self.rate_hz = rate_hz
        self._recording = False

    def connect(self) -> None:
        return None

    def calibrate(self) -> None:
        return None

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def poll(self, t: float) -> GazeSample | None:
        if not self._recording:
            return None
        x = (t * 200.0) % self.width
        y = self.height / 2 + 20.0 * math.sin(t * 2.0)
        return GazeSample(t=t, x=x, y=y, valid=True)

    def close(self) -> None:
        return None


_EDF_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{0,7}")


def _validate_edf_name(name: str) -> str:
    """Return a legal EyeLink Host PC data-file name, or raise ``ValueError``.

    pylink's ``openDataFile`` documents an MS-DOS style name: at most eight
    characters, using only ASCII letters, digits and the underscore, and the
    EyeLink library refuses an underscore in first place. Checking this before a
    session starts gives a clear error rather than a failure on the Host PC. An
    optional ``.edf`` suffix is accepted and normalised.
    """
    stem = name[:-4] if name.lower().endswith(".edf") else name
    if not _EDF_STEM.fullmatch(stem):
        raise ValueError(
            "EDF name must be 1 to 8 ASCII letters, digits or underscores, not "
            f"starting with an underscore, optionally with a .edf suffix. Got {name!r}"
        )
    return f"{stem}.edf"


def _gaze_from_components(
    x: float,
    y: float,
    time_ms: float,
    t0_ms: float = 0.0,
    *,
    missing: float = _MISSING_DATA,
) -> GazeSample:
    """Build a :class:`GazeSample` from raw EyeLink sample components.

    Gaze is reported in the display's pixel coordinates with a top-left origin and
    y pointing down, which is the convention the interest areas use, so no axis
    flip is needed. A coordinate equal to ``missing`` marks a blink or track loss
    and yields ``valid=False``. ``time_ms`` and ``t0_ms`` are Host PC times in
    milliseconds, and the sample time is returned in seconds after ``t0_ms``.
    The backend passes no ``t0_ms``, so the time stays on the Host PC clock and
    ``t * 1000`` is the sample's timestamp in the EDF file. This is pure, so the
    time base and the validity rule are tested without the device.
    """
    valid = x != missing and y != missing
    return GazeSample(t=(time_ms - t0_ms) / 1000.0, x=float(x), y=float(y), valid=valid)


class EyeLinkTracker:
    """Backend for SR Research EyeLink trackers over pylink.

    It follows the documented pylink call sequence for configuration,
    calibration, link sampling and the data-file transfer. The connection,
    trial, recording and closing calls run in pylink's dummy mode, and a test
    exercises them where pylink is installed. Calibration, the drift check and
    recording with a tracker have not yet been validated. The pure helpers are
    tested without pylink.

    pylink is imported lazily in :meth:`connect`, so importing this module never
    requires the SDK. The Host PC is reached over a dedicated Ethernet link at
    ``address`` through pylink's own transport rather than through Python sockets,
    so an active ``NetworkGuard`` does not block it. That link carries local
    hardware traffic, not internet traffic.

    The session runs each passage as one trial. :meth:`begin_trial` writes the
    passage's interest-area set beside the local copy of the EDF file and sends
    the ``TRIALID`` and ``!V IAREA FILE`` messages, and :meth:`end_trial` sends
    the trial variables and ``TRIAL_RESULT``, so EyeLink Data Viewer segments the
    recording by passage and loads the session's word boxes (see
    :mod:`readsync.dataviewer`). Recording runs during each passage only.

    Parameters
    ----------
    address:
        The Host PC address. The SR Research default is ``100.1.1.1``.
    sample_rate:
        Samples per second requested from the tracker, for example 1000.
    calibration:
        The calibration layout, for example ``HV9`` for nine points or ``HV5``.
    screen_size:
        Display size in pixels. It sets the tracker's coordinate system so gaze is
        reported in the same pixels as the interest areas.
    edf_name:
        Name of the raw data file written on the Host PC and fetched on
        :meth:`close`. Validated against the Host PC filename limit.
    receive_to:
        Local path for the fetched data file. Defaults to ``edf_name`` in the
        working directory.
    graphics:
        A ready-made calibration graphics environment. When neither this nor
        ``graphics_factory`` is given, pylink's built-in graphics are used.
    graphics_factory:
        Called with the connected pylink ``EyeLink`` object to build the
        calibration graphics, because SR Research's PsychoPy graphics class needs
        that object. To calibrate in the experiment's own window, pass
        ``lambda link: EyeLinkCoreGraphicsPsychoPy(link, presenter.window)``,
        with the class from SR Research's PsychoPy examples.
    dummy:
        Connect in no-tracker simulation mode for a wiring check with no hardware.
    interest_area_files:
        Write each trial's interest-area set beside the local EDF file and name it
        in the trial's messages. Set to ``False`` to send the trial messages alone.
    eye:
        The eye whose gaze drives the session's word and region events:
        ``"left"``, ``"right"`` or ``None``. With ``None``, a binocular sample
        uses the right eye and a monocular one the eye recorded. With a named
        eye, a sample without that eye's data is invalid. The data file keeps
        every recorded eye either way.
    """

    def __init__(
        self,
        *,
        address: str = "100.1.1.1",
        sample_rate: int = 1000,
        calibration: str = "HV9",
        screen_size: tuple[int, int] = (1920, 1080),
        edf_name: str = "readsync",
        receive_to: str | None = None,
        graphics: object | None = None,
        graphics_factory: Callable[[Any], object] | None = None,
        dummy: bool = False,
        interest_area_files: bool = True,
        eye: str | None = None,
    ) -> None:
        if eye not in (None, "left", "right"):
            raise ValueError(f"eye must be 'left', 'right' or None, got {eye!r}")
        self.address = address
        self.sample_rate = sample_rate
        self.calibration = calibration
        self.screen_size = screen_size
        self.edf_name = _validate_edf_name(edf_name)
        self.receive_to = receive_to
        self.graphics = graphics
        self.graphics_factory = graphics_factory
        self.dummy = dummy
        self.interest_area_files = interest_area_files
        self.eye = eye
        self._pylink: Any = None
        self._link: Any = None
        self._missing: float = _MISSING_DATA
        self._graphics_open = False
        self.last_calibration_message: str | None = None

    def connect(self) -> None:
        try:
            import pylink
        except ImportError as exc:
            raise RuntimeError(
                "pylink is not installed. Install the SR Research SDK on the lab "
                "machine, or use NullTracker for development."
            ) from exc
        self._configure(pylink)  # pragma: no cover - requires the EyeLink host

    def _configure(self, pylink: Any) -> None:  # pragma: no cover - requires the host
        self._pylink = pylink
        self._missing = float(pylink.MISSING_DATA)
        self._link = pylink.EyeLink(None if self.dummy else self.address)
        # SR Research's examples put the tracker in offline (idle) mode before
        # opening the data file and sending configuration commands.
        self._link.setOfflineMode()
        self._link.openDataFile(self.edf_name)
        width, height = self.screen_size
        coords = f"0 0 {width - 1} {height - 1}"
        self._link.sendCommand(f"sample_rate {self.sample_rate}")
        self._link.sendCommand(f"calibration_type = {self.calibration}")
        self._link.sendCommand(f"screen_pixel_coords = {coords}")
        self._link.sendMessage(f"DISPLAY_COORDS {coords}")
        self._link.sendCommand("recording_parse_type = GAZE")
        self._link.sendCommand(
            "file_event_filter = "
            "LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT"
        )
        # Pupil area and gaze-resolution channels are excluded as a
        # data-minimisation default, so the record holds gaze coordinates and
        # status alone. Re-enable them per protocol where a study needs them.
        self._link.sendCommand("file_sample_data = LEFT,RIGHT,GAZE,STATUS")
        self._link.sendCommand("link_sample_data = LEFT,RIGHT,GAZE,STATUS")

    def calibrate(self) -> None:
        if self._link is None:
            raise RuntimeError("connect() must be called before calibrate()")
        if not self._graphics_open:
            graphics = self.graphics
            if graphics is None and self.graphics_factory is not None:
                graphics = self.graphics_factory(self._link)
            if graphics is not None:
                self._pylink.openGraphicsEx(graphics)
            else:
                self._pylink.openGraphics(self.screen_size)
            self._graphics_open = True
        self._link.doTrackerSetup()
        # The Host PC reports the last calibration or validation outcome as a
        # text message. Keeping it lets the session log it and the per-session
        # quality report include it.
        self.last_calibration_message = str(self._link.getCalibrationMessage())

    def drift_correct(self) -> float | None:  # pragma: no cover - requires the host
        """Run a drift check at screen centre and return the reported error.

        The session calls this before each passage, while the tracker is idle.
        The result is a check: ``applyDriftCorrect`` is not called, so the
        calibration is left unchanged. ``doDriftCorrect`` returns a non-zero
        status when the check was aborted or escalated to a full recalibration,
        in which case no error value applies and ``None`` is returned. Otherwise
        the first number in the Host PC's calibration message is returned as
        the error; the message format, and the unit of that number, still have
        to be confirmed against the device.
        """
        if self._link is None:
            return None
        width, height = self.screen_size
        status = self._link.doDriftCorrect(width // 2, height // 2, 1, 1)
        if status != 0:
            # The check was aborted or escalated into a full recalibration, so
            # refresh the stored outcome for the session to log.
            self.last_calibration_message = str(self._link.getCalibrationMessage())
            return None
        message = str(self._link.getCalibrationMessage())
        for token in message.replace("=", " ").split():
            try:
                return float(token)
            except ValueError:
                continue
        return None

    def start_recording(self) -> None:  # pragma: no cover - requires the host
        if self._link is None:
            raise RuntimeError("connect() must be called before start_recording()")
        self._link.setOfflineMode()
        error = self._link.startRecording(1, 1, 1, 1)
        if error:
            raise RuntimeError(f"EyeLink startRecording failed with code {error}")
        self._pylink.beginRealTimeMode(100)
        # Let a few samples arrive before the passage is drawn.
        self._pylink.pumpDelay(100)

    def poll(self, t: float) -> GazeSample | None:
        if self._link is None:
            return None
        sample = self._link.getNewestSample()
        if sample is None:
            return None
        right = self.eye in (None, "right") and sample.isRightSample()
        left = self.eye in (None, "left") and sample.isLeftSample()
        if right:
            gaze_x, gaze_y = sample.getRightEye().getGaze()
        elif left:
            gaze_x, gaze_y = sample.getLeftEye().getGaze()
        else:
            gaze_x = gaze_y = self._missing
        return _gaze_from_components(
            gaze_x, gaze_y, float(sample.getTime()), missing=self._missing
        )

    def send_message(self, text: str) -> None:  # pragma: no cover - requires the host
        """Write a timestamped message into the data file for synchronisation.

        The Host PC stamps the message in the same clock as the gaze samples, so a
        message is how an on-screen event is aligned with the eye record. Route
        session markers here with :class:`readsync.markers.EyeLinkMarkerSink`.
        """
        if self._link is not None:
            self._link.sendMessage(text)

    def begin_trial(self, trial_id: str, areas: Sequence[InterestArea]) -> None:
        """Open a trial in the data file and attach its interest areas.

        Called by the session before the passage's drift check. The messages
        are written whether or not the tracker is recording, as in SR Research's
        trial structure.
        """
        ias_file = None
        if self.interest_area_files:
            local_edf = Path(self.receive_to or self.edf_name)
            ias_file = ias_filename(trial_id)
            write_ias(areas, local_edf.parent / ias_file)
        for message in trial_start_messages(trial_id, ias_file):
            self.send_message(message)

    def end_trial(self, variables: Mapping[str, object] | None = None) -> None:
        """Attach the trial variables and close the trial in the data file."""
        for message in trial_end_messages(variables):
            self.send_message(message)

    def stop_recording(self) -> None:  # pragma: no cover - requires the host
        if self._link is None:
            return
        self._pylink.pumpDelay(100)
        self._link.stopRecording()
        self._pylink.endRealTimeMode()

    def close(self) -> None:  # pragma: no cover - requires the host
        if self._link is None:
            return
        self._link.setOfflineMode()
        self._pylink.msecDelay(500)
        self._link.closeDataFile()
        if not self.dummy:
            self._link.receiveDataFile(self.edf_name, self.receive_to or self.edf_name)
        self._link.close()
        self._link = None
        if self._graphics_open:
            self._pylink.closeGraphics()
            self._graphics_open = False

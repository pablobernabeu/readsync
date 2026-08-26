"""Eye-tracker backends behind a small common interface.

The science needs a research-grade infrared tracker, but the toolkit is written
against a ``Tracker`` protocol so that the experiment logic does not depend on a
particular device, and so that it runs on a developer's laptop and in continuous
integration with no hardware at all.

``NullTracker`` produces synthetic gaze and is used for development and tests.
``EyeLinkTracker`` sketches the SR Research ``pylink`` integration, following the
documented call sequence, configuration and data flow, so the lab build is a
matter of installing the SDK, supplying calibration graphics and validating
against the device. The SDK is imported lazily, so importing this module never
requires it. New backends (for example a Tobii backend via Titta, or the webcam
tracker in ``readsync.webcam``) implement the same protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["GazeSample", "Tracker", "NullTracker", "EyeLinkTracker"]

# The EyeLink reports a coordinate equal to this value when an eye is lost, for
# example during a blink. pylink exposes it as ``pylink.MISSING_DATA``. It is
# repeated here so the validity logic can be tested without the SDK.
_MISSING_DATA = -32768.0


@dataclass(frozen=True)
class GazeSample:
    """One gaze estimate. ``t`` is seconds from recording start; ``x``/``y`` are
    pixels; ``valid`` is false during blinks or track loss."""

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


def _validate_edf_name(name: str) -> str:
    """Return a legal EyeLink Host PC data-file name, or raise ``ValueError``.

    The Host PC stores the raw data file under an 8.3-style name, so the stem must
    be eight characters or fewer and use only letters, digits or an underscore.
    Checking this before a session starts gives a clear error rather than a
    cryptic failure on the Host PC. An optional ``.edf`` suffix is accepted and
    normalised.
    """
    stem = name[:-4] if name.lower().endswith(".edf") else name
    if not stem or len(stem) > 8 or not all(c.isalnum() or c == "_" for c in stem):
        raise ValueError(
            "EDF name must be 1 to 8 letters, digits or underscores, optionally "
            f"with a .edf suffix. Got {name!r}"
        )
    return f"{stem}.edf"


def _gaze_from_components(
    x: float,
    y: float,
    time_ms: float,
    t0_ms: float,
    *,
    missing: float = _MISSING_DATA,
) -> GazeSample:
    """Build a :class:`GazeSample` from raw EyeLink sample components.

    Gaze is reported in the display's pixel coordinates with a top-left origin and
    y pointing down, which is the convention the interest areas use, so no axis
    flip is needed. A coordinate equal to ``missing`` marks a blink or track loss
    and yields ``valid=False``. ``time_ms`` and ``t0_ms`` are Host PC times in
    milliseconds, and the sample time is returned in seconds from the start of
    recording. This is pure, so the time base and the validity rule are tested
    without the device.
    """
    valid = x != missing and y != missing
    return GazeSample(t=(time_ms - t0_ms) / 1000.0, x=float(x), y=float(y), valid=valid)


class EyeLinkTracker:
    """Research-grade backend over the SR Research ``pylink`` SDK.

    This sketches the device integration from the documented ``pylink`` call
    sequence: the configuration, the calibration handshake, the link sampling
    and the data-file transfer. Completing it in the lab means installing the
    SDK, supplying a calibration graphics environment and validating against
    the hardware. Every line that touches the device is marked as not covered,
    because it cannot run without an EyeLink Host PC. The pure helpers it
    relies on are tested.

    pylink is imported lazily in :meth:`connect`, so importing this module never
    requires the SDK. The Host PC is reached over a dedicated Ethernet link at
    ``address`` through pylink's own transport rather than through Python sockets,
    so an active ``NetworkGuard`` does not block it. That link carries local
    hardware traffic, not internet traffic.

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
        The calibration graphics environment. The lab build passes an
        ``EyeLinkCoreGraphicsPsychoPy`` instance so calibration is drawn in the
        same PsychoPy window as the experiment. When ``None``, the SDK's built-in
        graphics are used.
    dummy:
        Connect in no-tracker simulation mode for a wiring check with no hardware.
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
        dummy: bool = False,
    ) -> None:
        self.address = address
        self.sample_rate = sample_rate
        self.calibration = calibration
        self.screen_size = screen_size
        self.edf_name = _validate_edf_name(edf_name)
        self.receive_to = receive_to
        self.graphics = graphics
        self.dummy = dummy
        self._pylink: Any = None
        self._link: Any = None
        self._missing: float = _MISSING_DATA
        self._t0_ms: float = 0.0
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

    def calibrate(self) -> None:  # pragma: no cover - requires the host
        if self._link is None:
            raise RuntimeError("connect() must be called before calibrate()")
        if self.graphics is not None:
            self._pylink.openGraphicsEx(self.graphics)
        else:
            self._pylink.openGraphics(self.screen_size)
        self._link.doTrackerSetup()
        # The Host PC reports the last calibration or validation outcome as a
        # text message. Keeping it lets the session log it and the per-session
        # quality report include it.
        self.last_calibration_message = str(self._link.getCalibrationMessage())

    def drift_correct(self) -> float | None:  # pragma: no cover - requires the host
        """Run a drift correction at screen centre and return the error in degrees.

        The session calls this between passages where the tracker offers it.
        ``doDriftCorrect`` returns a non-zero status when the check was aborted
        or escalated to a full recalibration, in which case no error value
        applies and ``None`` is returned. The error magnitude is read back from
        the Host PC's calibration message.
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
        error = self._link.startRecording(1, 1, 1, 1)
        if error:
            raise RuntimeError(f"EyeLink startRecording failed with code {error}")
        self._pylink.beginRealTimeMode(100)
        self._pylink.pumpDelay(100)
        self._t0_ms = float(self._link.trackerTime())

    def poll(self, t: float) -> GazeSample | None:  # pragma: no cover - requires the host
        if self._link is None:
            return None
        sample = self._link.getNewestSample()
        if sample is None:
            return None
        if sample.isRightSample():
            eye = sample.getRightEye()
        elif sample.isLeftSample():
            eye = sample.getLeftEye()
        else:
            return None
        gaze_x, gaze_y = eye.getGaze()
        return _gaze_from_components(
            gaze_x, gaze_y, float(sample.getTime()), self._t0_ms, missing=self._missing
        )

    def send_message(self, text: str) -> None:  # pragma: no cover - requires the host
        """Write a timestamped message into the data file for synchronisation.

        The Host PC stamps the message in the same clock as the gaze samples, so a
        message is how an on-screen event is aligned with the eye record. Route
        session markers here with :class:`readsync.markers.EyeLinkMarkerSink`.
        """
        if self._link is not None:
            self._link.sendMessage(text)

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

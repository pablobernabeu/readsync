"""A webcam eye-tracking backend behind the same ``Tracker`` interface.

This wires a camera and a gaze estimator into the
:class:`readsync.trackers.Tracker` protocol, so a session can run on a webcam
with only the tracker swapped. Landmark detection comes from MediaPipe's
FaceLandmarker, while the gaze features and the calibration maps are implemented
here. ``GazeEstimator`` and ``FrameSource`` are narrow protocols, so another
estimator or camera can be used instead. The tracker's own logic, which is rate
limiting, confidence gating and pixel scaling, runs without a camera.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..trackers import GazeSample
from ._linalg import ridge_fit

# The MediaPipe FaceLandmarker model is fetched once from Google's model store and
# kept locally, so a recording session needs no network. This is the canonical URL.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

__all__ = [
    "FrameSource",
    "GazeEstimator",
    "OpenCVCamera",
    "WebcamTracker",
    "Calibration",
    "SignalModel",
    "AffineCalibration",
    "PolynomialCalibration",
    "GazeSignals",
    "head_pose_from_matrix",
    "MediaPipeIrisEstimator",
]


@runtime_checkable
class FrameSource(Protocol):
    """A source of camera frames. ``read`` returns the next frame or ``None`` when
    one is not available. ``close`` releases the device."""

    def read(self) -> Any | None: ...
    def close(self) -> None: ...


@runtime_checkable
class GazeEstimator(Protocol):
    """Estimates gaze from a single frame.

    ``estimate`` returns ``(nx, ny, confidence)`` where ``nx`` and ``ny`` are the
    gaze position as fractions of the screen in ``[0, 1]`` and ``confidence`` is
    in ``[0, 1]``. It returns ``None`` when no gaze can be estimated, for example
    when no face is found or the estimator has not been calibrated. Mapping the
    eye to screen coordinates needs a per-participant calibration, which is the
    estimator's responsibility.
    """

    def estimate(self, frame: Any) -> tuple[float, float, float] | None: ...


def _to_pixels(nx: float, ny: float, screen_size: tuple[int, int]) -> tuple[float, float]:
    """Scale a normalised gaze position to pixels, clamped to the screen."""
    width, height = screen_size
    cx = min(max(nx, 0.0), 1.0)
    cy = min(max(ny, 0.0), 1.0)
    return cx * (width - 1), cy * (height - 1)


class OpenCVCamera:
    """A frame source backed by OpenCV ``VideoCapture``.

    OpenCV is optional and imported only when the camera is opened, so this
    module imports without it. OpenCV captures in BGR order. Frames are converted
    to RGB, the order MediaPipe expects, unless ``rgb`` is false. Reading a frame
    needs a camera, so this path is not covered by the test suite.
    """

    def __init__(self, device_index: int = 0, *, rgb: bool = True) -> None:
        self.device_index = device_index
        self.rgb = rgb
        self._cap: Any = None
        self._cv2: Any = None

    def _ensure_cap(self) -> Any:  # pragma: no cover - needs a camera
        if self._cap is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV (cv2) is not available. Install the 'webcam' extra "
                    '(pip install "readsync[webcam]") to capture from a camera.'
                ) from exc
            self._cv2 = cv2
            self._cap = cv2.VideoCapture(self.device_index)
        return self._cap

    def read(self) -> Any | None:  # pragma: no cover - needs a camera
        ok, frame = self._ensure_cap().read()
        if not ok:
            return None
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB) if self.rgb else frame

    def close(self) -> None:  # pragma: no cover - needs a camera
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamTracker:
    """Eye-tracker backend that drives a camera and a gaze estimator.

    It implements the :class:`readsync.trackers.Tracker` protocol, so it is passed
    to a :class:`~readsync.session.ReadingSession` in place of the infrared
    tracker with no other change. It emits at most ``max_rate_hz`` samples a
    second, which reflects the low frame rate of a webcam, and it marks a sample
    invalid when the estimator's confidence is below ``min_confidence``, when the
    estimate falls off the screen or when no gaze is returned. Each sample is
    stamped with the time the session polled it, so the delay between the
    camera's exposure and the estimate is not recorded. The camera read and the
    estimate run inside the session's display loop, so a slow estimate can
    delay the next frame.

    Parameters
    ----------
    estimator:
        The gaze model, behind the :class:`GazeEstimator` protocol.
    camera:
        The frame source. When ``None``, an :class:`OpenCVCamera` is opened on
        :meth:`connect`.
    screen_size:
        Display size in pixels, used to scale normalised gaze to the same pixels
        as the interest areas.
    max_rate_hz:
        The cap on emitted samples per second.
    min_confidence:
        The confidence at or above which a sample is treated as valid.
    device_index:
        Camera index for the default :class:`OpenCVCamera`.
    """

    def __init__(
        self,
        *,
        estimator: GazeEstimator,
        camera: FrameSource | None = None,
        screen_size: tuple[int, int] = (1920, 1080),
        max_rate_hz: float = 30.0,
        min_confidence: float = 0.5,
        device_index: int = 0,
    ) -> None:
        if max_rate_hz <= 0:
            raise ValueError("max_rate_hz must be positive")
        self.estimator = estimator
        self.screen_size = screen_size
        self.min_confidence = min_confidence
        self.device_index = device_index
        self._camera = camera
        self._min_interval = 1.0 / max_rate_hz
        self._recording = False
        self._last_emit: float | None = None

    def connect(self) -> None:
        if self._camera is None:  # pragma: no cover - needs a camera
            self._camera = OpenCVCamera(self.device_index)

    def calibrate(self) -> None:
        # Calibration maps the eye to the screen and is the estimator's
        # responsibility, since it depends on the model. It is delegated when the
        # estimator provides it, and is otherwise a no-op here.
        calibrate = getattr(self.estimator, "calibrate", None)
        if callable(calibrate):
            calibrate()

    def start_recording(self) -> None:
        self._recording = True
        self._last_emit = None

    def stop_recording(self) -> None:
        self._recording = False

    def poll(self, t: float) -> GazeSample | None:
        if not self._recording or self._camera is None:
            return None
        if self._last_emit is not None and (t - self._last_emit) < self._min_interval:
            return None
        frame = self._camera.read()
        if frame is None:
            return None
        self._last_emit = t
        estimate = self.estimator.estimate(frame)
        if estimate is None:
            return GazeSample(t=t, x=0.0, y=0.0, valid=False)
        nx, ny, confidence = estimate
        x, y = _to_pixels(nx, ny, self.screen_size)
        # An estimate off the screen is clamped to the edge for the record but
        # marked invalid, so it cannot land on a word near the edge.
        on_screen = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        return GazeSample(t=t, x=x, y=y, valid=confidence >= self.min_confidence and on_screen)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()


def _solve_3x3(matrix: list[list[float]], rhs: list[float]) -> tuple[float, float, float]:
    """Solve a 3 by 3 linear system by Gaussian elimination with partial pivoting."""
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("calibration points are degenerate; vary the targets")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(3):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            a[r] = [a[r][k] - factor * a[col][k] for k in range(4)]
    return a[0][3] / a[0][0], a[1][3] / a[1][1], a[2][3] / a[2][2]


def _fit_axis(
    design: Sequence[tuple[float, float, float]], target: Sequence[float]
) -> tuple[float, float, float]:
    """Least-squares coefficients for one output axis via the normal equations."""
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    rhs = [0.0, 0.0, 0.0]
    for row, value in zip(design, target, strict=True):
        for i in range(3):
            rhs[i] += row[i] * value
            for j in range(3):
                matrix[i][j] += row[i] * row[j]
    return _solve_3x3(matrix, rhs)


@runtime_checkable
class Calibration(Protocol):
    """Maps a two-dimensional eye feature to a normalised screen position."""

    def map(self, feature: tuple[float, float]) -> tuple[float, float]: ...


class SignalModel(Protocol):
    """Maps one frame's gaze signals to a normalised screen position.

    :class:`readsync.webcam.gaze.HeadAwareGazeModel` implements it.
    """

    def map(self, signals: GazeSignals) -> tuple[float, float]: ...


class AffineCalibration:
    """A two-dimensional affine map from an eye feature to normalised gaze.

    Webcam gaze needs a per-participant calibration: a short routine where the
    reader looks at known screen positions while the eye feature is recorded, then
    a map is fitted from feature to screen. This fits the affine map
    ``(nx, ny) = A (fx, fy) + b`` by ordinary least squares, the simplest such
    map. It does not compensate for head movement after calibration.
    :class:`PolynomialCalibration` and :class:`readsync.webcam.gaze.HeadAwareGazeModel`
    are richer alternatives.
    """

    def __init__(
        self,
        coeffs_x: tuple[float, float, float],
        coeffs_y: tuple[float, float, float],
    ) -> None:
        self._cx = coeffs_x
        self._cy = coeffs_y

    @classmethod
    def fit(
        cls,
        features: Sequence[tuple[float, float]],
        targets: Sequence[tuple[float, float]],
    ) -> AffineCalibration:
        """Fit the map from recorded ``features`` to known screen ``targets``."""
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")
        if len(features) < 3:
            raise ValueError("at least three calibration points are required")
        design = [(fx, fy, 1.0) for fx, fy in features]
        coeffs_x = _fit_axis(design, [tx for tx, _ in targets])
        coeffs_y = _fit_axis(design, [ty for _, ty in targets])
        return cls(coeffs_x, coeffs_y)

    def map(self, feature: tuple[float, float]) -> tuple[float, float]:
        """Map an eye feature to a normalised screen position."""
        fx, fy = feature
        ax, bx, cx = self._cx
        ay, by, cy = self._cy
        return ax * fx + bx * fy + cx, ay * fx + by * fy + cy


class PolynomialCalibration:
    """A polynomial map from an eye feature to normalised gaze.

    The relationship between iris position and screen position is not linear, so an
    affine map leaves systematic error, most of it near the edges of the screen.
    A low-order polynomial can absorb some of that curvature. The default is
    second order, with a small ridge penalty that keeps the fit stable when
    calibration points are few or noisy.
    """

    def __init__(
        self,
        coeffs_x: Sequence[float],
        coeffs_y: Sequence[float],
        degree: int,
    ) -> None:
        self._cx = list(coeffs_x)
        self._cy = list(coeffs_y)
        self._degree = degree

    @staticmethod
    def _terms(fx: float, fy: float, degree: int) -> list[float]:
        """Monomials ``fx**i * fy**j`` for all ``i + j <= degree``."""
        return [
            fx**i * fy ** (total - i)
            for total in range(degree + 1)
            for i in range(total + 1)
        ]

    @classmethod
    def fit(
        cls,
        features: Sequence[tuple[float, float]],
        targets: Sequence[tuple[float, float]],
        *,
        degree: int = 2,
        ridge: float = 1e-3,
    ) -> PolynomialCalibration:
        """Fit the polynomial map from ``features`` to known screen ``targets``."""
        if len(features) != len(targets):
            raise ValueError("features and targets must have the same length")
        design = [cls._terms(fx, fy, degree) for fx, fy in features]
        n_terms = len(design[0])
        if len(features) < n_terms and ridge <= 0:
            raise ValueError(
                f"need at least {n_terms} points for degree {degree}, or set ridge > 0"
            )
        coeffs_x = ridge_fit(design, [tx for tx, _ in targets], ridge)
        coeffs_y = ridge_fit(design, [ty for _, ty in targets], ridge)
        return cls(coeffs_x, coeffs_y, degree)

    def map(self, feature: tuple[float, float]) -> tuple[float, float]:
        """Map an eye feature to a normalised screen position."""
        terms = self._terms(feature[0], feature[1], self._degree)
        nx = sum(c * t for c, t in zip(self._cx, terms, strict=True))
        ny = sum(c * t for c, t in zip(self._cy, terms, strict=True))
        return nx, ny


@dataclass(frozen=True)
class GazeSignals:
    """Per-frame signals for head-pose-aware gaze estimation.

    ``fx`` and ``fy`` are the iris position within the eye, in the eye's own axes. ``yaw``,
    ``pitch`` and ``roll`` are the head orientation in radians. ``openness`` is the
    eye-aspect ratio, which falls towards zero during a blink.
    """

    fx: float
    fy: float
    yaw: float
    pitch: float
    roll: float
    openness: float


def head_pose_from_matrix(matrix: Any) -> tuple[float, float, float]:
    """Extract ``(yaw, pitch, roll)`` in radians from a 4x4 head-pose matrix.

    Reads the rotation block of the FaceLandmarker facial transformation matrix and
    decomposes it as intrinsic z-y-x angles. The exact convention matters only for
    naming. The three angles serve as monotone head-orientation inputs and measure
    how far the head leaves the calibrated range. A near-degenerate matrix falls
    back to a zero roll instead of raising.
    """
    r = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    sy = math.sqrt(r[0][0] * r[0][0] + r[1][0] * r[1][0])
    if sy > 1e-6:
        roll = math.atan2(r[1][0], r[0][0])
        yaw = math.atan2(-r[2][0], sy)
        pitch = math.atan2(r[2][1], r[2][2])
    else:
        roll = 0.0
        yaw = math.atan2(-r[2][0], sy)
        pitch = math.atan2(-r[1][2], r[1][1])
    return yaw, pitch, roll


class MediaPipeIrisEstimator:
    """A :class:`GazeEstimator` over MediaPipe's FaceLandmarker.

    FaceLandmarker returns 478 face landmarks, the last ten of which are the
    irises, and a facial transformation matrix that gives the head pose. Each
    frame is reduced to :class:`GazeSignals`: the iris position within the eye,
    the head's yaw, pitch and roll and an eye-openness ratio. A fitted ``model``,
    such as :class:`readsync.webcam.gaze.HeadAwareGazeModel`, maps all the
    signals to the screen and so compensates for head pose. Without one, a
    ``calibration`` maps the iris feature alone. With neither, the estimator
    returns ``None``, because raw iris positions are not screen coordinates.

    Confidence is 0 during a blink, when the openness falls below
    ``blink_openness``, and when the head is more than ``max_pose_drift_deg``
    outside the range seen at calibration, as reported by a model that offers
    ``pose_drift_degrees``. Otherwise, it is 1. The openness is computed from
    landmark coordinates normalised by the frame's width and height, so the same
    threshold corresponds to different eye closures on cameras with different
    aspect ratios. The vertical iris feature is divided by the lid gap, so a
    partly closed eye that stays above the threshold shifts the estimate
    downwards. :class:`WebcamTracker` marks
    low-confidence samples invalid, so the session holds the current word and
    region through them.

    FaceLandmarker needs a model file. Download it once from
    :data:`FACE_LANDMARKER_URL` and pass its path as ``model_path``. After that,
    a session needs no network. mediapipe is optional and imported only when the
    first frame is processed. Frames must be RGB ``uint8`` arrays, which is what
    :class:`OpenCVCamera` returns by default. The landmark path needs a camera
    and is not covered by the test suite, while :meth:`estimate_from_signals`
    is.
    """

    # Landmark indices in the 478-point model, as
    # ``(iris, corner, corner, upper lid, lower lid)`` for each eye. The feature is
    # measured in the eye's own axes, along the corner-to-corner line and its
    # perpendicular.
    _RIGHT_EYE = (468, 33, 133, 159, 145)
    _LEFT_EYE = (473, 362, 263, 386, 374)

    def __init__(
        self,
        *,
        calibration: Calibration | None = None,
        model: SignalModel | None = None,
        model_path: str = "face_landmarker.task",
        num_faces: int = 1,
        blink_openness: float = 0.15,
        max_pose_drift_deg: float | None = 12.0,
    ) -> None:
        self.calibration: Calibration | None = calibration
        self.model: SignalModel | None = model
        self.model_path = model_path
        self.num_faces = num_faces
        self.blink_openness = blink_openness
        self.max_pose_drift_deg = max_pose_drift_deg
        self._landmarker: Any = None
        self._mp: Any = None

    def _ensure_landmarker(self) -> Any:  # pragma: no cover - needs mediapipe and a model
        if self._landmarker is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
            except ImportError as exc:
                raise RuntimeError(
                    "mediapipe is not installed. Install the 'webcam' extra "
                    '(pip install "readsync[webcam]") to use the webcam estimator.'
                ) from exc
            path = Path(self.model_path)
            if not path.is_file():
                raise RuntimeError(
                    f"FaceLandmarker model not found at {path}. Download it once "
                    f"from {FACE_LANDMARKER_URL} and pass its path as model_path."
                )
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                num_faces=self.num_faces,
                output_facial_transformation_matrixes=True,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._mp = mp
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    @staticmethod
    def _eye_feature(landmarks: Any, spec: tuple[int, int, int, int, int]) -> tuple[float, float]:
        """Iris position in the eye's own frame, as fractions of width and height.

        The horizontal value is the iris projected onto the corner-to-corner axis
        and the vertical value is its projection onto the perpendicular, each
        normalised by the matching span. On a square frame, the feature stays the
        same when the head and eyes roll together. It changes when the head rolls while the gaze
        stays on one point, because the eyes then turn within the head. The
        landmark x and y are fractions of the frame's width and height, so on a
        non-square frame these lengths and directions are distorted by the
        aspect ratio. A span of exactly zero is replaced with 1e-6 to avoid
        dividing by zero, and implausible features from a nearly closed eye are
        left for the caller to reject.
        """
        iris, corner_a, corner_b, upper, lower = spec
        ax, ay = landmarks[corner_a].x, landmarks[corner_a].y
        bx, by = landmarks[corner_b].x, landmarks[corner_b].y
        ix, iy = landmarks[iris].x, landmarks[iris].y
        axis_x, axis_y = bx - ax, by - ay
        width = math.hypot(axis_x, axis_y) or 1e-6
        ux, uy = axis_x / width, axis_y / width  # unit vector along the eye
        fx = ((ix - ax) * ux + (iy - ay) * uy) / width
        cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
        height = (
            math.hypot(
                landmarks[lower].x - landmarks[upper].x,
                landmarks[lower].y - landmarks[upper].y,
            )
            or 1e-6
        )
        fy = ((ix - cx) * -uy + (iy - cy) * ux) / height  # perpendicular projection
        return fx, fy

    @staticmethod
    def _eye_openness(landmarks: Any, spec: tuple[int, int, int, int, int]) -> float:
        """Eye-aspect ratio (lid gap over eye width), which falls towards zero in a blink."""
        _, corner_a, corner_b, upper, lower = spec
        width = (
            math.hypot(
                landmarks[corner_b].x - landmarks[corner_a].x,
                landmarks[corner_b].y - landmarks[corner_a].y,
            )
            or 1e-6
        )
        height = math.hypot(
            landmarks[lower].x - landmarks[upper].x,
            landmarks[lower].y - landmarks[upper].y,
        )
        return height / width

    def feature(
        self, frame: Any
    ) -> tuple[float, float] | None:  # pragma: no cover - needs a camera
        signals = self.signals(frame)
        return None if signals is None else (signals.fx, signals.fy)

    def signals(self, frame: Any) -> GazeSignals | None:  # pragma: no cover - needs a camera
        """Read the per-frame gaze signals: iris feature, head pose and openness."""
        landmarker = self._ensure_landmarker()
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        rx, ry = self._eye_feature(landmarks, self._RIGHT_EYE)
        lx, ly = self._eye_feature(landmarks, self._LEFT_EYE)
        openness = (
            self._eye_openness(landmarks, self._RIGHT_EYE)
            + self._eye_openness(landmarks, self._LEFT_EYE)
        ) / 2.0
        if result.facial_transformation_matrixes:
            yaw, pitch, roll = head_pose_from_matrix(result.facial_transformation_matrixes[0])
        else:
            yaw = pitch = roll = 0.0
        return GazeSignals((rx + lx) / 2.0, (ry + ly) / 2.0, yaw, pitch, roll, openness)

    def estimate_from_signals(self, signals: GazeSignals) -> tuple[float, float, float] | None:
        """Map one frame's signals to ``(nx, ny, confidence)``."""
        if self.model is not None:
            nx, ny = self.model.map(signals)
        elif self.calibration is not None:
            nx, ny = self.calibration.map((signals.fx, signals.fy))
        else:
            return None
        confidence = 1.0
        if signals.openness < self.blink_openness:
            confidence = 0.0
        drift = getattr(self.model, "pose_drift_degrees", None)
        if (
            self.max_pose_drift_deg is not None
            and drift is not None
            and float(drift(signals)) > self.max_pose_drift_deg
        ):
            confidence = 0.0
        return nx, ny, confidence

    def estimate(
        self, frame: Any
    ) -> tuple[float, float, float] | None:  # pragma: no cover - needs a camera
        if self.model is None and self.calibration is None:
            return None
        signals = self.signals(frame)
        return None if signals is None else self.estimate_from_signals(signals)

"""A webcam eye-tracking backend behind the same ``Tracker`` interface.

This wires a camera and a gaze estimator into the
:class:`readsync.trackers.Tracker` protocol, so a session can run on a webcam
exactly as it runs on the infrared tracker, with nothing else changed. It is the
apparatus for the methods strand, never a substitute for the research-grade
device. Webcam gaze is far coarser in space and slower in time, so it cannot
support word-level reading measures. The subpackage docstring states where it is
nonetheless useful and what the strand aims to improve.

The estimation comes from existing tooling and is not reimplemented here.
``GazeEstimator`` and ``FrameSource`` are narrow protocols, so a published model
plugs in behind them. The tracker's own logic, which is rate limiting, confidence
gating and pixel scaling, is pure and is tested without a camera.
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
    estimator's responsibility, exactly as in established browser tooling.
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

    OpenCV is optional and imported lazily, so this module imports without it.
    Reading a frame needs a camera, so the read path runs in the lab and not in
    continuous integration. Frames are returned in OpenCV's BGR order, so an
    estimator that expects RGB must convert them.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cap: Any = None

    def _ensure_cap(self) -> Any:  # pragma: no cover - needs a camera
        if self._cap is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "OpenCV (cv2) is not available. Install the 'webcam' extra "
                    '(pip install "readsync[webcam]") to capture from a camera.'
                ) from exc
            self._cap = cv2.VideoCapture(self.device_index)
        return self._cap

    def read(self) -> Any | None:  # pragma: no cover - needs a camera
        ok, frame = self._ensure_cap().read()
        return frame if ok else None

    def close(self) -> None:  # pragma: no cover - needs a camera
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class WebcamTracker:
    """Eye-tracker backend that drives a camera and a gaze estimator.

    It implements the :class:`readsync.trackers.Tracker` protocol, so it is passed
    to a :class:`~readsync.session.ReadingSession` in place of the infrared
    tracker with no other change. Two properties keep it explicit about webcam
    limits. It emits at most ``max_rate_hz`` samples a second, which reflects the
    low frame rate of a webcam, and it marks a sample invalid when the estimator's
    confidence is below ``min_confidence`` or when no gaze is returned.

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
        return GazeSample(t=t, x=x, y=y, valid=confidence >= self.min_confidence)

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


class AffineCalibration:
    """A two-dimensional affine map from an eye feature to normalised gaze.

    Webcam gaze needs a per-participant calibration: a short routine where the
    reader looks at known screen positions while the eye feature is recorded, then
    a map is fitted from feature to screen. This fits the affine map
    ``(nx, ny) = A (fx, fy) + b`` by ordinary least squares, the simplest map that
    corrects for head position and camera placement. It is pure Python and is
    tested. A richer regression, for example a ridge or a second-order polynomial
    fit, can replace it behind the same :meth:`map` call.
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
    Fitting a low-order polynomial removes much of that, which is the usual way to
    improve webcam gaze accuracy. The default is second order. A small ridge
    penalty keeps the fit stable when calibration points are few or noisy, which is
    why this is preferred over an unregularised fit here. Pure Python and tested.
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

    ``fx`` and ``fy`` are the roll-invariant iris position within the eye. ``yaw``,
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
    naming; the three angles serve as monotone head-orientation inputs and to
    measure drift from the calibration pose. A near-degenerate matrix falls back to
    a zero roll rather than raising.
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
    """Gaze features from the MediaPipe FaceLandmarker iris landmarks.

    The FaceLandmarker model returns 478 landmarks, the last ten of which are the
    irises, and the iris position within the eye is a usable gaze feature. This
    adapter turns that established model into a :class:`GazeEstimator`: it reads
    the iris offset and maps it to the screen with a fitted
    :class:`AffineCalibration`. The model is the existing tool, and only the thin
    adapter and the calibration live here.

    The current mediapipe wheels expose the Tasks API rather than the older
    ``solutions`` modules, so this uses ``FaceLandmarker``, which needs a model
    file. Download it once from :data:`FACE_LANDMARKER_URL` and pass its path as
    ``model_path``. After that a session needs no network.

    mediapipe is optional and imported lazily, so this module imports without it.
    The landmark path needs a camera and is exercised in the example and the lab,
    not in continuous integration. Frames must be RGB ``uint8`` arrays. Without a
    calibration the estimator returns ``None``, because raw iris offsets are not
    screen coordinates.
    """

    # Landmark indices in the 478-point model, as
    # ``(iris, corner, corner, upper lid, lower lid)`` for each eye. The feature is
    # measured in the eye's own axes (along the corner-to-corner line and its
    # perpendicular), which makes it invariant to head roll.
    _RIGHT_EYE = (468, 33, 133, 159, 145)
    _LEFT_EYE = (473, 362, 263, 386, 374)

    def __init__(
        self,
        *,
        calibration: Calibration | None = None,
        model_path: str = "face_landmarker.task",
        num_faces: int = 1,
    ) -> None:
        self.calibration: Calibration | None = calibration
        self.model_path = model_path
        self.num_faces = num_faces
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
        normalised by the matching span. Measuring in the eye's own axes makes the
        feature invariant to head roll. Pure landmark arithmetic, so it is
        unit-tested without a camera. A small floor on each span avoids a divide by
        zero when the eye is closed.
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
        """Eye-aspect ratio (lid gap over eye width). Falls to zero during a blink."""
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

    def estimate(
        self, frame: Any
    ) -> tuple[float, float, float] | None:  # pragma: no cover - needs a camera
        if self.calibration is None:
            return None
        feature = self.feature(frame)
        if feature is None:
            return None
        nx, ny = self.calibration.map(feature)
        return nx, ny, 1.0

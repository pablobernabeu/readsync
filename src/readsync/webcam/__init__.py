"""Webcam gaze estimation.

:class:`WebcamTracker` drives a camera and a gaze estimator behind the same
``Tracker`` interface as the EyeLink backend, so a session can run on a webcam.
:class:`MediaPipeIrisEstimator` takes face and iris landmarks and the head pose
from MediaPipe's FaceLandmarker, and :mod:`readsync.webcam.gaze` fits a
regularised, head-pose-aware calibration from those signals to the screen.
:mod:`readsync.webcam.benchmark` converts the error between paired gaze points to
degrees of visual angle, for comparison with a reference eye-tracker.

Webcam gaze is coarser and noisier than infrared eye-tracking. In one comparison
with an EyeLink 1000, a webcam system erred by 1.45 degrees on average against
0.91 for the EyeLink (Kaduk et al., 2024, https://doi.org/10.3758/s13428-023-02237-8),
and a systematic evaluation of WebGazer on a laptop found errors of about 10 to 14
degrees (Psarra et al., 2026, https://doi.org/10.3390/jemr19050099). This
module's own accuracy has not yet been measured against a reference tracker.
"""

from __future__ import annotations

from .benchmark import angular_error, mean_angular_error, pixels_to_degrees
from .calibration import (
    average_feature,
    fit_from_fixations,
    fit_polynomial_from_fixations,
    median_feature,
)
from .gaze import (
    HeadAwareGazeModel,
    aggregate_signals,
    cross_validated_error,
    fit_head_aware,
)
from .tracker import (
    FACE_LANDMARKER_URL,
    AffineCalibration,
    Calibration,
    FrameSource,
    GazeEstimator,
    GazeSignals,
    MediaPipeIrisEstimator,
    OpenCVCamera,
    PolynomialCalibration,
    WebcamTracker,
    head_pose_from_matrix,
)

__all__ = [
    "WebcamTracker",
    "GazeEstimator",
    "FrameSource",
    "OpenCVCamera",
    "Calibration",
    "AffineCalibration",
    "PolynomialCalibration",
    "GazeSignals",
    "head_pose_from_matrix",
    "HeadAwareGazeModel",
    "aggregate_signals",
    "fit_head_aware",
    "cross_validated_error",
    "MediaPipeIrisEstimator",
    "FACE_LANDMARKER_URL",
    "average_feature",
    "median_feature",
    "fit_from_fixations",
    "fit_polynomial_from_fixations",
    "pixels_to_degrees",
    "angular_error",
    "mean_angular_error",
]

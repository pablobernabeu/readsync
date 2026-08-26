"""Webcam eye-tracking benchmark and improvement (research strand).

This subpackage holds the toolkit's methods strand: an open benchmark of
webcam and camera-based gaze estimation on reading materials, and targeted
improvements to calibration for text.

The current evidence is that webcam gaze is too coarse spatially, with roughly
one and a half to ten degrees of error, and too slow temporally, at about thirty
to forty samples a second, for word-level reading measures. It cannot substitute
for the infrared tracker. The aim of this strand is to map where webcam methods
are good enough, coarse screening for example, and to push that boundary,
releasing the results and code openly.

The strand builds on existing tooling for webcam gaze instead of reimplementing
it. :class:`WebcamTracker` drives a camera and a pluggable gaze estimator behind
the same ``Tracker`` interface the rest of the toolkit uses, and
:mod:`readsync.webcam.benchmark` reports accuracy in degrees against the infrared
tracker on the same readers and materials.
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

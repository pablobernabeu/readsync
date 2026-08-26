"""Fitting a webcam gaze calibration from fixation samples.

A webcam calibration shows the reader a few targets at known screen positions and
records the eye feature while they fixate each one. Averaging the feature over the
fixation reduces noise, and an affine map is then fitted from the averaged
features to the target positions. These helpers are pure and are tested. The
interactive part that shows the targets and reads the camera lives in the example
script, because it needs a display and a camera.
"""

from __future__ import annotations

from collections.abc import Sequence

from .tracker import AffineCalibration, PolynomialCalibration

__all__ = [
    "average_feature",
    "median_feature",
    "fit_from_fixations",
    "fit_polynomial_from_fixations",
]


def average_feature(features: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Return the mean of a set of two-dimensional eye features."""
    if not features:
        raise ValueError("need at least one feature to average")
    n = len(features)
    return sum(f[0] for f in features) / n, sum(f[1] for f in features) / n


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def median_feature(features: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Return the per-axis median of a set of features.

    The median ignores the occasional stray frame, for example a half-blink during
    a fixation, so it is a steadier centre for a calibration target than the mean.
    """
    if not features:
        raise ValueError("need at least one feature to take the median of")
    return _median([f[0] for f in features]), _median([f[1] for f in features])


def fit_from_fixations(
    fixations: Sequence[Sequence[tuple[float, float]]],
    targets: Sequence[tuple[float, float]],
) -> AffineCalibration:
    """Fit a calibration from per-target fixation samples.

    ``fixations`` holds, for each target, the eye features recorded while the
    reader fixated it. ``targets`` holds the matching screen positions as
    fractions of the screen in ``[0, 1]``. Each target's features are averaged,
    then the affine map is fitted. At least three targets are needed, and they
    must not be collinear.
    """
    if len(fixations) != len(targets):
        raise ValueError("fixations and targets must have the same length")
    averaged = [average_feature(samples) for samples in fixations]
    return AffineCalibration.fit(averaged, targets)


def fit_polynomial_from_fixations(
    fixations: Sequence[Sequence[tuple[float, float]]],
    targets: Sequence[tuple[float, float]],
    *,
    degree: int = 2,
    ridge: float = 1e-3,
) -> PolynomialCalibration:
    """Fit a regularised polynomial calibration from per-target fixation samples.

    Each target's features are reduced to their median, then a polynomial map is
    fitted. This captures the curvature of the iris-to-screen relationship that an
    affine map cannot, which is the main lever on webcam gaze accuracy.
    """
    if len(fixations) != len(targets):
        raise ValueError("fixations and targets must have the same length")
    centres = [median_feature(samples) for samples in fixations]
    return PolynomialCalibration.fit(centres, targets, degree=degree, ridge=ridge)

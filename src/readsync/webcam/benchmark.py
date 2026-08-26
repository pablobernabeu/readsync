"""Compare two gaze streams in angular terms, for the webcam methods strand.

The decisive measure of a webcam tracker is its error in degrees of visual angle
against a research-grade tracker recording the same reader on the same materials.
Pixel error alone is not comparable across screens and seating distances, so it
is converted to an angle with the screen geometry. These helpers are pure and are
tested.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["pixels_to_degrees", "angular_error", "mean_angular_error"]

Point = tuple[float, float]


def pixels_to_degrees(pixels: float, *, viewing_distance_mm: float, px_per_mm: float) -> float:
    """Convert a distance on screen in pixels to degrees of visual angle.

    ``viewing_distance_mm`` is the eye-to-screen distance and ``px_per_mm`` is the
    display's pixel density. The exact arctangent is used, so the result is
    correct for large eccentricities as well as small ones.
    """
    if viewing_distance_mm <= 0 or px_per_mm <= 0:
        raise ValueError("viewing_distance_mm and px_per_mm must be positive")
    millimetres = pixels / px_per_mm
    return math.degrees(math.atan2(millimetres, viewing_distance_mm))


def angular_error(a: Point, b: Point, *, viewing_distance_mm: float, px_per_mm: float) -> float:
    """Angular distance in degrees between two gaze points given in screen pixels."""
    distance = math.hypot(a[0] - b[0], a[1] - b[1])
    return pixels_to_degrees(
        distance, viewing_distance_mm=viewing_distance_mm, px_per_mm=px_per_mm
    )


def mean_angular_error(
    reference: Sequence[Point],
    estimate: Sequence[Point],
    *,
    viewing_distance_mm: float,
    px_per_mm: float,
) -> float:
    """Mean angular error in degrees between paired reference and estimate points.

    ``reference`` is the research-grade gaze and ``estimate`` is the webcam gaze,
    sampled at matched moments. The two sequences must be the same length.
    """
    if len(reference) != len(estimate):
        raise ValueError("reference and estimate must have the same length")
    if not reference:
        raise ValueError("at least one paired point is required")
    total = sum(
        angular_error(r, e, viewing_distance_mm=viewing_distance_mm, px_per_mm=px_per_mm)
        for r, e in zip(reference, estimate, strict=True)
    )
    return total / len(reference)

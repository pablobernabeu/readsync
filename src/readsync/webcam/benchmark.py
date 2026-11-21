"""Compare two gaze streams in degrees of visual angle.

A webcam tracker's accuracy is best expressed as its error in degrees of visual
angle against a research-grade tracker recording the same reader on the same
materials. Pixel error is not comparable across screens and seating distances,
so it is converted to an angle with the screen geometry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["pixels_to_degrees", "angular_error", "mean_angular_error"]

Point = tuple[float, float]


def pixels_to_degrees(pixels: float, *, viewing_distance_mm: float, px_per_mm: float) -> float:
    """Convert an on-screen extent in pixels to degrees of visual angle.

    The extent is measured from the point on the screen straight ahead of the
    eye. ``viewing_distance_mm`` is the eye-to-screen distance and ``px_per_mm``
    is the display's pixel density. For the angle between two arbitrary screen
    points use :func:`angular_error`, which accounts for where they lie.
    """
    if viewing_distance_mm <= 0 or px_per_mm <= 0:
        raise ValueError("viewing_distance_mm and px_per_mm must be positive")
    millimetres = pixels / px_per_mm
    return math.degrees(math.atan2(millimetres, viewing_distance_mm))


def angular_error(
    a: Point,
    b: Point,
    *,
    viewing_distance_mm: float,
    px_per_mm: float,
    eye_px: Point,
) -> float:
    """Angle in degrees between the lines of sight to two screen points.

    ``a``, ``b`` and ``eye_px`` are in screen pixels, where ``eye_px`` is the
    point on the screen straight ahead of the eye, usually the screen centre.
    The angle is taken between the two lines of sight in three dimensions, so it
    holds wherever the points lie on the screen. Converting their pixel
    separation as if it were centred on the line of sight would overstate the
    error away from the centre.
    """
    if viewing_distance_mm <= 0 or px_per_mm <= 0:
        raise ValueError("viewing_distance_mm and px_per_mm must be positive")

    def sight_line(point: Point) -> tuple[float, float, float]:
        return (
            (point[0] - eye_px[0]) / px_per_mm,
            (point[1] - eye_px[1]) / px_per_mm,
            viewing_distance_mm,
        )

    ux, uy, uz = sight_line(a)
    vx, vy, vz = sight_line(b)
    cross = math.sqrt(
        (uy * vz - uz * vy) ** 2 + (uz * vx - ux * vz) ** 2 + (ux * vy - uy * vx) ** 2
    )
    dot = ux * vx + uy * vy + uz * vz
    return math.degrees(math.atan2(cross, dot))


def mean_angular_error(
    reference: Sequence[Point],
    estimate: Sequence[Point],
    *,
    viewing_distance_mm: float,
    px_per_mm: float,
    eye_px: Point,
) -> float:
    """Mean angular error in degrees between paired reference and estimate points.

    ``reference`` is the research-grade gaze and ``estimate`` is the webcam gaze,
    sampled at matched moments. Webcam samples logged by a session carry the
    time they were polled, not the camera's exposure time, so pairing them by
    time leaves an unmeasured latency. The two sequences must be the same
    length.
    ``eye_px`` is as in :func:`angular_error`.
    """
    if len(reference) != len(estimate):
        raise ValueError("reference and estimate must have the same length")
    if not reference:
        raise ValueError("at least one paired point is required")
    total = sum(
        angular_error(
            r, e, viewing_distance_mm=viewing_distance_mm, px_per_mm=px_per_mm, eye_px=eye_px
        )
        for r, e in zip(reference, estimate, strict=True)
    )
    return total / len(reference)

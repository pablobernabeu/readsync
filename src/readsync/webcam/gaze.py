"""Head-pose-aware gaze model for the webcam strand.

The iris feature alone is confounded by head orientation, so even a good
polynomial calibration drifts when the head turns. This model adds the head yaw
and pitch as inputs, so the calibration can compensate for the head pose the
reader actually held, and it records how far the head has moved from the
calibration pose, so gaze can be flagged when the reader has shifted too far to
trust.

The fit is a regularised linear least squares over an explicit design that mixes a
second-order polynomial in the iris feature with head-pose terms and their products
with the feature, so head pose can rescale the feature, not only shift it.
The maths is pure Python and is tested; only reading the signals from a frame needs
a camera.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._linalg import ridge_fit
from .tracker import GazeSignals

__all__ = [
    "HeadAwareGazeModel",
    "aggregate_signals",
    "fit_head_aware",
    "cross_validated_error",
]

# Twelve design terms: a second-order polynomial in the iris feature, linear head
# yaw and pitch, and the products of head pose with the iris feature. The pose-by-
# feature products let head pose change the gain of the feature, not only add an
# offset, which is what a still-head calibration held at two poses needs in order to
# fit both poses instead of averaging them. Roll is handled by the roll-invariant
# iris feature itself.
_N_TERMS = 12


def _design(signals: GazeSignals) -> list[float]:
    fx, fy = signals.fx, signals.fy
    yaw, pitch = signals.yaw, signals.pitch
    return [
        1.0, fx, fy, fx * fx, fy * fy, fx * fy, yaw, pitch,
        yaw * fx, yaw * fy, pitch * fx, pitch * fy,
    ]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def aggregate_signals(samples: Sequence[GazeSignals]) -> GazeSignals:
    """Reduce a target's signal samples to their per-field median.

    The median ignores the occasional stray frame, for example a half-blink during
    a fixation, so it is a steadier centre than the mean.
    """
    if not samples:
        raise ValueError("need at least one signal sample to aggregate")
    return GazeSignals(
        _median([s.fx for s in samples]),
        _median([s.fy for s in samples]),
        _median([s.yaw for s in samples]),
        _median([s.pitch for s in samples]),
        _median([s.roll for s in samples]),
        _median([s.openness for s in samples]),
    )


@dataclass(frozen=True)
class HeadAwareGazeModel:
    """Maps gaze signals to a normalised screen position, compensating head pose.

    ``pose_lo`` and ``pose_hi`` are the per-axis range of head orientation seen
    during calibration. Calibrating at more than one head position widens that
    range and gives the yaw and pitch terms something to learn from, so the model
    compensates across the calibrated range, not only near a single pose.
    """

    coeffs_x: list[float]
    coeffs_y: list[float]
    pose_lo: tuple[float, float, float]
    pose_hi: tuple[float, float, float]

    @classmethod
    def fit(
        cls,
        signals: Sequence[GazeSignals],
        targets: Sequence[tuple[float, float]],
        *,
        ridge: float = 1e-3,
    ) -> HeadAwareGazeModel:
        """Fit the model from per-target signals to known screen targets."""
        if len(signals) != len(targets):
            raise ValueError("signals and targets must have the same length")
        if len(signals) < _N_TERMS and ridge <= 0:
            raise ValueError(f"need at least {_N_TERMS} calibration points, or set ridge > 0")
        design = [_design(s) for s in signals]
        coeffs_x = ridge_fit(design, [tx for tx, _ in targets], ridge)
        coeffs_y = ridge_fit(design, [ty for _, ty in targets], ridge)
        yaws = [s.yaw for s in signals]
        pitches = [s.pitch for s in signals]
        rolls = [s.roll for s in signals]
        pose_lo = (min(yaws), min(pitches), min(rolls))
        pose_hi = (max(yaws), max(pitches), max(rolls))
        return cls(coeffs_x, coeffs_y, pose_lo, pose_hi)

    def map(self, signals: GazeSignals) -> tuple[float, float]:
        """Predict the normalised screen position for one frame's signals."""
        terms = _design(signals)
        nx = sum(c * v for c, v in zip(self.coeffs_x, terms, strict=True))
        ny = sum(c * v for c, v in zip(self.coeffs_y, terms, strict=True))
        return nx, ny

    def pose_drift_degrees(self, signals: GazeSignals) -> float:
        """How far the head is outside the calibrated pose range, in degrees.

        Zero while the head stays within the range seen at calibration, growing
        once it moves beyond it, which is where the compensation stops being
        supported by data and the gaze should no longer be trusted.
        """

        def outside(value: float, low: float, high: float) -> float:
            return max(0.0, low - value, value - high)

        worst = max(
            outside(signals.yaw, self.pose_lo[0], self.pose_hi[0]),
            outside(signals.pitch, self.pose_lo[1], self.pose_hi[1]),
            outside(signals.roll, self.pose_lo[2], self.pose_hi[2]),
        )
        return math.degrees(worst)


def fit_head_aware(
    per_target_samples: Sequence[Sequence[GazeSignals]],
    targets: Sequence[tuple[float, float]],
    *,
    ridge: float = 1e-3,
) -> HeadAwareGazeModel:
    """Aggregate each target's samples, then fit a head-aware model."""
    if len(per_target_samples) != len(targets):
        raise ValueError("samples and targets must have the same length")
    centres = [aggregate_signals(samples) for samples in per_target_samples]
    return HeadAwareGazeModel.fit(centres, targets, ridge=ridge)


def cross_validated_error(
    centres: Sequence[GazeSignals],
    targets: Sequence[tuple[float, float]],
    *,
    ridge: float = 1e-3,
    groups: Sequence[int] | None = None,
) -> float:
    """Leave-one-group-out mean error in normalised screen units.

    For each group the model is fitted on the others and then used to predict the
    held-out group's points. The mean predicted-to-true distance is an unbiased
    accuracy estimate, because no point is scored by a model that saw it. With
    multi-pose calibration, ``groups`` labels each point with its screen target, so
    a target's samples at every head pose are held out together and a pose seen for
    that target elsewhere cannot leak in. Without ``groups`` each point is its own
    group, the plain leave-one-out case.
    """
    n = len(centres)
    if n != len(targets):
        raise ValueError("centres and targets must have the same length")
    labels = list(range(n)) if groups is None else list(groups)
    if len(labels) != n:
        raise ValueError("groups must have one label per point")
    folds = sorted(set(labels))
    if len(folds) < 3:
        raise ValueError("need at least three groups for cross-validation")
    total = 0.0
    count = 0
    for held in folds:
        train_signals = [centres[j] for j in range(n) if labels[j] != held]
        train_targets = [targets[j] for j in range(n) if labels[j] != held]
        model = HeadAwareGazeModel.fit(train_signals, train_targets, ridge=max(ridge, 1e-6))
        for j in range(n):
            if labels[j] == held:
                px, py = model.map(centres[j])
                tx, ty = targets[j]
                total += math.hypot(px - tx, py - ty)
                count += 1
    return total / count

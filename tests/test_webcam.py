"""Tests for the webcam strand.

The camera and the gaze model are behind protocols, so the tracker's own logic is
tested with scripted stand-ins and needs no hardware. The calibration fit and the
angular-error helpers are pure maths and are checked against known values.
"""

from __future__ import annotations

import math

import pytest

from readsync.webcam import (
    AffineCalibration,
    GazeSignals,
    HeadAwareGazeModel,
    PolynomialCalibration,
    WebcamTracker,
    aggregate_signals,
    angular_error,
    average_feature,
    cross_validated_error,
    fit_from_fixations,
    fit_head_aware,
    fit_polynomial_from_fixations,
    head_pose_from_matrix,
    mean_angular_error,
    median_feature,
    pixels_to_degrees,
)


class ScriptedCamera:
    """A frame source that yields a sentinel frame a fixed number of times."""

    def __init__(self, frames: int = 1000) -> None:
        self._left = frames
        self.closed = False

    def read(self) -> object | None:
        if self._left <= 0:
            return None
        self._left -= 1
        return object()

    def close(self) -> None:
        self.closed = True


class ScriptedEstimator:
    """A gaze estimator that returns a fixed normalised point and confidence."""

    def __init__(self, point: tuple[float, float, float] | None) -> None:
        self.point = point
        self.calibrated = False

    def estimate(self, frame: object) -> tuple[float, float, float] | None:
        return self.point

    def calibrate(self) -> None:
        self.calibrated = True


def _running_tracker(estimator: ScriptedEstimator, **kwargs: object) -> WebcamTracker:
    tracker = WebcamTracker(
        estimator=estimator,
        camera=ScriptedCamera(),
        screen_size=(1000, 800),
        **kwargs,  # type: ignore[arg-type]
    )
    tracker.connect()
    tracker.start_recording()
    return tracker


def test_poll_is_silent_until_recording() -> None:
    tracker = WebcamTracker(estimator=ScriptedEstimator((0.5, 0.5, 1.0)), camera=ScriptedCamera())
    assert tracker.poll(0.0) is None


def test_poll_scales_normalised_gaze_to_pixels() -> None:
    tracker = _running_tracker(ScriptedEstimator((0.5, 0.25, 0.9)))
    sample = tracker.poll(0.0)
    assert sample is not None
    assert sample.x == pytest.approx(0.5 * 999)
    assert sample.y == pytest.approx(0.25 * 799)
    assert sample.valid


def test_low_confidence_is_marked_invalid() -> None:
    tracker = _running_tracker(ScriptedEstimator((0.5, 0.5, 0.2)), min_confidence=0.5)
    sample = tracker.poll(0.0)
    assert sample is not None and not sample.valid


def test_no_gaze_yields_an_invalid_sample() -> None:
    tracker = _running_tracker(ScriptedEstimator(None))
    sample = tracker.poll(0.0)
    assert sample is not None and not sample.valid


def test_emission_is_throttled_to_the_max_rate() -> None:
    tracker = _running_tracker(ScriptedEstimator((0.5, 0.5, 1.0)), max_rate_hz=10.0)
    assert tracker.poll(0.0) is not None  # first sample emits
    assert tracker.poll(0.05) is None  # 0.05 s later is inside the 0.1 s interval
    assert tracker.poll(0.10) is not None  # a full interval on, it emits again


def test_calibrate_delegates_to_the_estimator() -> None:
    estimator = ScriptedEstimator((0.5, 0.5, 1.0))
    tracker = WebcamTracker(estimator=estimator, camera=ScriptedCamera())
    tracker.calibrate()
    assert estimator.calibrated


def test_close_releases_the_camera() -> None:
    camera = ScriptedCamera()
    tracker = WebcamTracker(estimator=ScriptedEstimator(None), camera=camera)
    tracker.close()
    assert camera.closed


def test_affine_calibration_recovers_a_known_map() -> None:
    # A known affine map from feature to normalised screen position.
    def true_map(fx: float, fy: float) -> tuple[float, float]:
        return 0.5 * fx + 0.1 * fy + 0.2, -0.3 * fx + 0.4 * fy + 0.05

    features = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.2)]
    targets = [true_map(fx, fy) for fx, fy in features]
    calibration = AffineCalibration.fit(features, targets)

    predicted = calibration.map((0.3, 0.7))
    expected = true_map(0.3, 0.7)
    assert predicted[0] == pytest.approx(expected[0], abs=1e-9)
    assert predicted[1] == pytest.approx(expected[1], abs=1e-9)


def test_affine_calibration_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        AffineCalibration.fit([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)])


def test_affine_calibration_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        AffineCalibration.fit([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])


def test_pixels_to_degrees_matches_the_arctangent() -> None:
    # 10 mm offset at 573 mm is close to one degree of visual angle.
    degrees = pixels_to_degrees(10.0, viewing_distance_mm=573.0, px_per_mm=1.0)
    assert degrees == pytest.approx(math.degrees(math.atan2(10.0, 573.0)))
    assert degrees == pytest.approx(1.0, abs=0.01)


def test_angular_error_and_mean() -> None:
    one = angular_error((0.0, 0.0), (10.0, 0.0), viewing_distance_mm=573.0, px_per_mm=1.0)
    assert one == pytest.approx(1.0, abs=0.01)

    mean = mean_angular_error(
        [(0.0, 0.0), (0.0, 0.0)],
        [(10.0, 0.0), (0.0, 10.0)],
        viewing_distance_mm=573.0,
        px_per_mm=1.0,
    )
    assert mean == pytest.approx(1.0, abs=0.01)


def test_mean_angular_error_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_angular_error([], [], viewing_distance_mm=573.0, px_per_mm=1.0)


def _quadratic(fx: float, fy: float) -> tuple[float, float]:
    return (
        0.2 + 0.5 * fx - 0.1 * fy + 0.3 * fx * fy - 0.2 * fx**2 + 0.05 * fy**2,
        0.1 - 0.2 * fx + 0.4 * fy + 0.1 * fx**2 - 0.05 * fx * fy + 0.2 * fy**2,
    )


def test_polynomial_calibration_recovers_a_quadratic_map() -> None:
    features = [(fx, fy) for fx in (0.0, 0.5, 1.0) for fy in (0.0, 0.5, 1.0)]
    targets = [_quadratic(*f) for f in features]
    calibration = PolynomialCalibration.fit(features, targets, degree=2, ridge=0.0)

    predicted = calibration.map((0.3, 0.7))
    expected = _quadratic(0.3, 0.7)
    assert predicted[0] == pytest.approx(expected[0], abs=1e-6)
    assert predicted[1] == pytest.approx(expected[1], abs=1e-6)


def test_polynomial_calibration_ridge_solves_with_few_points() -> None:
    # Four points are fewer than the six terms of a quadratic; ridge makes the
    # system solvable and keeps the fit close to the targets.
    features = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    targets = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    calibration = PolynomialCalibration.fit(features, targets, degree=2, ridge=1e-2)
    for feature, target in zip(features, targets, strict=True):
        mx, my = calibration.map(feature)
        assert mx == pytest.approx(target[0], abs=0.15)
        assert my == pytest.approx(target[1], abs=0.15)


def test_polynomial_calibration_underdetermined_without_ridge_raises() -> None:
    with pytest.raises(ValueError, match="at least"):
        PolynomialCalibration.fit(
            [(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)], degree=2, ridge=0.0
        )


def test_median_feature_is_per_axis_median() -> None:
    assert median_feature([(0.0, 1.0), (2.0, 3.0), (1.0, 5.0)]) == pytest.approx((1.0, 3.0))
    with pytest.raises(ValueError, match="median"):
        median_feature([])


def test_fit_polynomial_from_fixations_recovers_a_linear_map() -> None:
    features = [(fx, fy) for fx in (0.0, 0.5, 1.0) for fy in (0.0, 0.5, 1.0)]
    fixations = [[(fx - 0.01, fy), (fx + 0.01, fy)] for fx, fy in features]
    targets = [(0.1 + 0.8 * fx, 0.1 + 0.8 * fy) for fx, fy in features]
    calibration = fit_polynomial_from_fixations(fixations, targets, ridge=0.0)

    assert isinstance(calibration, PolynomialCalibration)
    mx, my = calibration.map((0.3, 0.7))
    assert mx == pytest.approx(0.1 + 0.8 * 0.3, abs=1e-6)
    assert my == pytest.approx(0.1 + 0.8 * 0.7, abs=1e-6)


class _Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _eye_landmarks(iris: tuple[float, float]) -> list[_Landmark]:
    """Build a 478-landmark list for the right eye with a given iris position."""
    from readsync.webcam.tracker import MediaPipeIrisEstimator

    _, corner_a, corner_b, upper, lower = MediaPipeIrisEstimator._RIGHT_EYE
    iris_idx = MediaPipeIrisEstimator._RIGHT_EYE[0]
    landmarks = [_Landmark(0.0, 0.0) for _ in range(478)]
    landmarks[corner_a] = _Landmark(0.40, 0.50)  # eye axis horizontal, width 0.10
    landmarks[corner_b] = _Landmark(0.50, 0.50)
    landmarks[upper] = _Landmark(0.45, 0.46)  # lid gap 0.08
    landmarks[lower] = _Landmark(0.45, 0.54)
    landmarks[iris_idx] = _Landmark(*iris)
    return landmarks


def _rotate(
    landmarks: list[_Landmark], angle: float, centre: tuple[float, float]
) -> list[_Landmark]:
    cos, sin = math.cos(angle), math.sin(angle)
    cx, cy = centre
    out = []
    for lm in landmarks:
        dx, dy = lm.x - cx, lm.y - cy
        out.append(_Landmark(cx + dx * cos - dy * sin, cy + dx * sin + dy * cos))
    return out


def test_eye_feature_values_in_eye_frame() -> None:
    from readsync.webcam.tracker import MediaPipeIrisEstimator

    landmarks = _eye_landmarks((0.46, 0.52))  # iris right of corner a and below centre
    fx, fy = MediaPipeIrisEstimator._eye_feature(landmarks, MediaPipeIrisEstimator._RIGHT_EYE)
    assert fx == pytest.approx(0.6)  # 0.06 along a 0.10-wide eye
    assert fy == pytest.approx(0.25)  # 0.02 below centre over a 0.08 lid gap


def test_eye_feature_is_roll_invariant() -> None:
    from readsync.webcam.tracker import MediaPipeIrisEstimator

    landmarks = _eye_landmarks((0.47, 0.52))
    upright = MediaPipeIrisEstimator._eye_feature(landmarks, MediaPipeIrisEstimator._RIGHT_EYE)
    rolled = _rotate(landmarks, math.radians(25.0), (0.45, 0.50))
    tilted = MediaPipeIrisEstimator._eye_feature(rolled, MediaPipeIrisEstimator._RIGHT_EYE)
    assert tilted[0] == pytest.approx(upright[0], abs=1e-9)
    assert tilted[1] == pytest.approx(upright[1], abs=1e-9)


def test_eye_openness_falls_when_the_lid_gap_shrinks() -> None:
    from readsync.webcam.tracker import MediaPipeIrisEstimator

    spec = MediaPipeIrisEstimator._RIGHT_EYE
    open_eye = _eye_landmarks((0.45, 0.50))
    open_ear = MediaPipeIrisEstimator._eye_openness(open_eye, spec)
    closing = _eye_landmarks((0.45, 0.50))
    closing[spec[3]] = _Landmark(0.45, 0.499)  # lids almost touching
    closing[spec[4]] = _Landmark(0.45, 0.501)
    assert MediaPipeIrisEstimator._eye_openness(closing, spec) < open_ear


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rotation_4x4(yaw: float, pitch: float, roll: float) -> list[list[float]]:
    cz, sz = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    r = _matmul(_matmul(rz, ry), rx)
    return [r[i] + [0.0] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _signal(
    fx: float, fy: float, yaw: float = 0.0, pitch: float = 0.0, openness: float = 0.3
) -> GazeSignals:
    return GazeSignals(fx, fy, yaw, pitch, 0.0, openness)


def _grid_signals() -> list[GazeSignals]:
    signals = []
    for i in range(16):
        fx, fy = (i % 4) / 3.0, (i // 4) / 3.0
        yaw, pitch = (((i * 7) % 5) - 2) / 10.0, (((i * 11) % 5) - 2) / 10.0
        signals.append(_signal(fx, fy, yaw, pitch))
    return signals


def test_head_pose_round_trips_through_the_matrix() -> None:
    yaw, pitch, roll = 0.2, -0.15, 0.1
    y, p, r = head_pose_from_matrix(_rotation_4x4(yaw, pitch, roll))
    assert y == pytest.approx(yaw, abs=1e-6)
    assert p == pytest.approx(pitch, abs=1e-6)
    assert r == pytest.approx(roll, abs=1e-6)


def test_head_aware_model_compensates_head_pose() -> None:
    def true_map(s: GazeSignals) -> tuple[float, float]:
        return 0.1 + 0.6 * s.fx + 0.3 * s.yaw, 0.1 + 0.6 * s.fy - 0.2 * s.pitch

    signals = _grid_signals()
    targets = [true_map(s) for s in signals]
    model = HeadAwareGazeModel.fit(signals, targets, ridge=1e-9)

    probe = _signal(0.3, 0.7, 0.15, -0.1)
    px, py = model.map(probe)
    ex, ey = true_map(probe)
    assert px == pytest.approx(ex, abs=1e-3)
    assert py == pytest.approx(ey, abs=1e-3)


def test_head_aware_model_fits_two_poses_with_different_gain() -> None:
    # Calibrate at two head positions, each held still (constant yaw and pitch), as
    # the app does. Head pose here changes the *gain* of the iris feature, not just
    # an offset, so an additive pose term cannot fit both poses and would adhere to
    # one of them. The pose-by-feature interaction terms fit each pose's mapping.
    def true_map(s: GazeSignals) -> tuple[float, float]:
        return (
            0.1 + (0.5 + 1.0 * s.yaw) * s.fx + 0.3 * s.yaw,
            0.1 + (0.5 - 1.0 * s.pitch) * s.fy - 0.2 * s.pitch,
        )

    axis = (0.15, 0.5, 0.85)
    signals = [
        _signal(fx, fy, yaw, pitch)
        for yaw, pitch in ((0.0, 0.1), (0.5, -0.2))
        for fy in axis
        for fx in axis
    ]
    model = HeadAwareGazeModel.fit(signals, [true_map(s) for s in signals], ridge=1e-6)

    # The model must be accurate at *both* calibrated poses, not only the last one.
    for yaw, pitch in ((0.0, 0.1), (0.5, -0.2)):
        probe = _signal(0.3, 0.7, yaw, pitch)
        px, py = model.map(probe)
        ex, ey = true_map(probe)
        assert px == pytest.approx(ex, abs=1e-2)
        assert py == pytest.approx(ey, abs=1e-2)


def test_cross_validated_error_is_small_for_consistent_data() -> None:
    signals = _grid_signals()
    targets = [(0.1 + 0.6 * s.fx + 0.3 * s.yaw, 0.1 + 0.6 * s.fy) for s in signals]
    assert cross_validated_error(signals, targets, ridge=1e-6) < 0.02


def test_pose_drift_is_zero_within_range_and_grows_outside() -> None:
    model = HeadAwareGazeModel(
        coeffs_x=[0.0] * 12,
        coeffs_y=[0.0] * 12,
        pose_lo=(math.radians(-5.0), 0.0, 0.0),
        pose_hi=(math.radians(5.0), 0.0, 0.0),
    )
    # Within the calibrated yaw range there is no drift.
    assert model.pose_drift_degrees(_signal(0.5, 0.5, yaw=0.0)) == pytest.approx(0.0)
    # 12 degrees of yaw is 7 degrees beyond the +5 degree edge of the range.
    drift = model.pose_drift_degrees(_signal(0.5, 0.5, yaw=math.radians(12.0)))
    assert drift == pytest.approx(7.0, abs=1e-6)


def test_grouped_cross_validation_holds_out_all_of_a_targets_poses() -> None:
    # Two head poses per target; the target depends on yaw, so the model must use
    # the yaw term. Grouping by target holds out both poses of a target together.
    signals: list[GazeSignals] = []
    targets: list[tuple[float, float]] = []
    groups: list[int] = []
    for pose_shift in (0.0, 0.12):
        for i, s in enumerate(_grid_signals()):
            signals.append(GazeSignals(s.fx, s.fy, s.yaw + pose_shift, s.pitch, 0.0, 0.3))
            targets.append((0.1 + 0.6 * s.fx + 0.3 * (s.yaw + pose_shift), 0.1 + 0.6 * s.fy))
            groups.append(i)
    error = cross_validated_error(signals, targets, ridge=1e-6, groups=groups)
    assert error < 0.05


def test_aggregate_signals_is_per_field_median() -> None:
    samples = [
        _signal(0.0, 1.0, openness=0.2),
        _signal(2.0, 3.0, openness=0.4),
        _signal(1.0, 5.0, openness=0.3),
    ]
    agg = aggregate_signals(samples)
    assert (agg.fx, agg.fy, agg.openness) == pytest.approx((1.0, 3.0, 0.3))


def test_fit_head_aware_aggregates_then_fits() -> None:
    targets = [(0.1 + 0.8 * ((i % 4) / 3.0), 0.1 + 0.8 * ((i // 4) / 3.0)) for i in range(16)]
    per_target = []
    for i in range(16):
        fx, fy = (i % 4) / 3.0, (i // 4) / 3.0
        per_target.append([_signal(fx - 0.01, fy), _signal(fx + 0.01, fy)])
    model = fit_head_aware(per_target, targets, ridge=1e-9)

    assert isinstance(model, HeadAwareGazeModel)
    mx, my = model.map(_signal(0.3, 0.7))
    assert mx == pytest.approx(0.1 + 0.8 * 0.3, abs=1e-3)
    assert my == pytest.approx(0.1 + 0.8 * 0.7, abs=1e-3)


def test_average_feature_is_the_mean() -> None:
    assert average_feature([(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]) == pytest.approx((1.0, 2.0))


def test_average_feature_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        average_feature([])


def test_fit_from_fixations_averages_then_fits() -> None:
    # A known affine map; each target sees a few noisy-looking but symmetric samples
    # that average back to the true feature.
    def true_map(fx: float, fy: float) -> tuple[float, float]:
        return 0.4 * fx + 0.2 * fy + 0.1, 0.1 * fx + 0.5 * fy + 0.05

    true_features = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    fixations = [[(fx - 0.01, fy), (fx + 0.01, fy)] for fx, fy in true_features]
    targets = [true_map(fx, fy) for fx, fy in true_features]

    calibration = fit_from_fixations(fixations, targets)
    predicted = calibration.map((0.3, 0.6))
    expected = true_map(0.3, 0.6)
    assert predicted[0] == pytest.approx(expected[0], abs=1e-6)
    assert predicted[1] == pytest.approx(expected[1], abs=1e-6)


def test_fit_from_fixations_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_from_fixations([[(0.0, 0.0)]], [(0.0, 0.0), (1.0, 1.0)])

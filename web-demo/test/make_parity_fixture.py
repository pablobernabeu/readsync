"""Generate the cross-language parity fixture for the browser gaze port.

The desktop implementation (readsync.webcam.gaze) is the reference. This script
builds a deterministic two-pose calibration, fits the head-aware model, and
records the fitted coefficients, mapped probe points, pose-drift values and the
cross-validated error into ``fixtures/parity.json``. It also records the eye
features and openness for sets of landmarks, and the head-pose angles for
matrices packed as MediaPipe packs them. The Node test suite runs the
JavaScript port on the same inputs and checks every output against these
numbers.

Regenerate after any deliberate change to the gaze maths, on either side:

    python test/make_parity_fixture.py

The fixture is deterministic (a fixed linear congruential sequence stands in
for random numbers), so regeneration is reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from readsync.webcam.gaze import (  # noqa: E402
    HeadAwareGazeModel,
    aggregate_signals,
    cross_validated_error,
)
from readsync.webcam.tracker import (  # noqa: E402
    GazeSignals,
    MediaPipeIrisEstimator,
    head_pose_from_matrix,
)

RIDGE = 1e-3


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % 2**31
        yield state / 2**31


def _rotation(yaw: float, pitch: float, roll: float) -> list[list[float]]:
    """A 4x4 transform whose rotation is Rz(roll) Ry(yaw) Rx(pitch)."""
    import math

    cz, sz = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]

    def mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    r = mul(mul(rz, ry), rx)
    return [r[0] + [0.0], r[1] + [0.0], r[2] + [0.0], [0.0, 0.0, 0.0, 1.0]]


def _head_pose_cases() -> list[dict[str, object]]:
    """Head-pose matrices packed as MediaPipe packs them, with the desktop angles."""
    cases = []
    for angles in [(0.2, -0.15, 0.1), (0.3, 0.25, 0.05), (-0.1, 0.05, -0.2)]:
        matrix = _rotation(*angles)
        packed = [matrix[i][j] for j in range(4) for i in range(4)]  # column-major
        cases.append({"packed": packed, "angles": list(head_pose_from_matrix(matrix))})
    return cases


def _eye_cases() -> list[dict[str, object]]:
    """Landmark sets with tilted, unequal eyes, and the desktop eye measures."""
    from types import SimpleNamespace

    noise = _lcg(7)
    cases = []
    for tilt in (0.0, 0.12, -0.2):
        points = [{"x": 0.0, "y": 0.0} for _ in range(478)]
        for spec, cx in ((MediaPipeIrisEstimator._RIGHT_EYE, 0.40),
                         (MediaPipeIrisEstimator._LEFT_EYE, 0.60)):
            iris, a, b, up, lo = spec
            for index, (dx, dy) in {a: (-0.05, 0.0), b: (0.05, 0.0), up: (0.0, -0.02),
                                    lo: (0.0, 0.021), iris: (0.012, -0.004)}.items():
                jitter = (next(noise) - 0.5) * 0.004
                x = cx + dx * (1 - tilt * tilt / 2) - dy * tilt + jitter
                y = 0.5 + dx * tilt + dy + jitter
                points[index] = {"x": x, "y": y}
        landmarks = [SimpleNamespace(**pt) for pt in points]
        used = sorted(set(MediaPipeIrisEstimator._RIGHT_EYE + MediaPipeIrisEstimator._LEFT_EYE))
        cases.append({
            "points": [[i, points[i]["x"], points[i]["y"]] for i in used],
            "right": list(MediaPipeIrisEstimator._eye_feature(
                landmarks, MediaPipeIrisEstimator._RIGHT_EYE)),
            "left": list(MediaPipeIrisEstimator._eye_feature(
                landmarks, MediaPipeIrisEstimator._LEFT_EYE)),
            "openness": [
                MediaPipeIrisEstimator._eye_openness(landmarks, MediaPipeIrisEstimator._RIGHT_EYE),
                MediaPipeIrisEstimator._eye_openness(landmarks, MediaPipeIrisEstimator._LEFT_EYE),
            ],
        })
    return cases


def _signal_cases(
    eyes: list[dict[str, object]], poses: list[dict[str, object]]
) -> dict[str, object]:
    """A full FaceLandmarker-shaped result and a median over several signals.

    The expected signals combine the two eyes and the head pose as
    MediaPipeIrisEstimator.signals does, after MediaPipe's Python API has
    unpacked the column-major matrix into rows.
    """
    results = []
    for eye, pose in zip(eyes, poses, strict=True):
        rx, ry = eye["right"]  # type: ignore[misc]
        lx, ly = eye["left"]  # type: ignore[misc]
        yaw, pitch, roll = pose["angles"]  # type: ignore[misc]
        openness = sum(eye["openness"]) / 2  # type: ignore[arg-type]
        results.append({
            "points": eye["points"],
            "packed": pose["packed"],
            "expected": [(rx + lx) / 2, (ry + ly) / 2, yaw, pitch, roll, openness],
        })
    noise = _lcg(11)
    samples = [
        GazeSignals(*(next(noise) for _ in range(6))) for _ in range(7)
    ]
    median = aggregate_signals(samples)
    return {
        "results": results,
        "median_samples": [vars(s) for s in samples],
        "median": vars(median),
    }


def build() -> dict[str, object]:
    noise = _lcg(20260826)
    signals: list[GazeSignals] = []
    targets: list[tuple[float, float]] = []
    groups: list[int] = []
    grid = [0.1, 0.5, 0.9]
    for pose, (yaw0, pitch0) in enumerate([(0.02, -0.01), (0.24, 0.09)]):
        group = 0
        for ny in grid:
            for nx in grid:
                jitter = next(noise) * 0.02
                signals.append(
                    GazeSignals(
                        fx=0.2 + 0.6 * nx + 0.05 * yaw0 + jitter * 0.1,
                        fy=-0.3 + 0.6 * ny + 0.08 * pitch0 - jitter * 0.05,
                        yaw=yaw0 + jitter * 0.01,
                        pitch=pitch0 - jitter * 0.02,
                        roll=0.01 * pose + jitter * 0.005,
                        openness=0.32 + jitter * 0.01,
                    )
                )
                targets.append((nx, ny))
                groups.append(group)
                group += 1
    model = HeadAwareGazeModel.fit(signals, targets, ridge=RIDGE)
    probes = [
        GazeSignals(fx=0.35, fy=-0.12, yaw=0.05, pitch=0.02, roll=0.0, openness=0.3),
        GazeSignals(fx=0.62, fy=0.18, yaw=0.30, pitch=0.12, roll=0.02, openness=0.28),
        GazeSignals(fx=0.15, fy=-0.35, yaw=-0.20, pitch=-0.15, roll=-0.03, openness=0.33),
    ]
    head_pose = _head_pose_cases()
    eyes = _eye_cases()
    return {
        "ridge": RIDGE,
        "signals": [vars(s) for s in signals],
        "targets": targets,
        "groups": groups,
        "coeffs_x": model.coeffs_x,
        "coeffs_y": model.coeffs_y,
        "pose_lo": model.pose_lo,
        "pose_hi": model.pose_hi,
        "probes": [vars(p) for p in probes],
        "mapped": [model.map(p) for p in probes],
        "drift_degrees": [model.pose_drift_degrees(p) for p in probes],
        "cv_error": cross_validated_error(signals, targets, ridge=RIDGE, groups=groups),
        "head_pose": head_pose,
        "eyes": eyes,
        "extraction": _signal_cases(eyes, head_pose),
    }


if __name__ == "__main__":
    out = HERE / "fixtures" / "parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

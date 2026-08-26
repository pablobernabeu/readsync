"""Generate the cross-language parity fixture for the browser gaze port.

The desktop implementation (readsync.webcam.gaze) is the reference. This script
builds a deterministic two-pose calibration, fits the head-aware model, and
records the fitted coefficients, mapped probe points, pose-drift values and the
cross-validated error into ``fixtures/parity.json``. The Node test suite then
fits the JavaScript port on the same inputs and checks every output against
these numbers, so the two implementations are held to the same arithmetic, not
merely to the same properties.

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
    cross_validated_error,
)
from readsync.webcam.tracker import GazeSignals  # noqa: E402

RIDGE = 1e-3


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % 2**31
        yield state / 2**31


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
    }


if __name__ == "__main__":
    out = HERE / "fixtures" / "parity.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

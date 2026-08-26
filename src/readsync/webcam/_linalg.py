"""Small dependency-free linear algebra for the calibration fits.

The webcam calibration solves modest least-squares systems, so a compact, tested
Gaussian-elimination solver is enough and keeps the core free of a numerical
dependency. numpy is present at runtime through the camera stack, but the fits
live here in pure Python so they import and test without it.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["solve", "ridge_fit"]


def solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a square linear system by Gaussian elimination with partial pivoting.

    Raises ``ValueError`` if the system is singular, which the callers turn into a
    request to vary the calibration targets or add a ridge penalty.
    """
    n = len(matrix)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("linear system is singular; vary the targets or add ridge")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            a[r] = [a[r][k] - factor * a[col][k] for k in range(n + 1)]
    return [a[i][n] / a[i][i] for i in range(n)]


def ridge_fit(
    design: Sequence[Sequence[float]], target: Sequence[float], ridge: float
) -> list[float]:
    """Ridge least-squares coefficients via the regularised normal equations.

    The penalty is not applied to the constant term at index 0, so a constant
    offset is never shrunk towards zero. A positive ``ridge`` also keeps the system
    solvable when the design has fewer rows than columns.
    """
    n = len(design[0])
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, value in zip(design, target, strict=True):
        for i in range(n):
            atb[i] += row[i] * value
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    for i in range(1, n):
        ata[i][i] += ridge
    return solve(ata, atb)

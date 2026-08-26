// Small dependency-free linear algebra for the calibration fits.
// A faithful port of readsync/src/readsync/webcam/_linalg.py, so the browser
// calibration behaves like the desktop one. Pure functions, unit-tested in Node.

export function solve(matrix, rhs) {
  // Solve a square linear system by Gaussian elimination with partial pivoting.
  const n = matrix.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    }
    if (Math.abs(a[pivot][col]) < 1e-12) {
      throw new Error("linear system is singular; vary the targets or add ridge");
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      for (let k = 0; k <= n; k++) a[r][k] -= factor * a[col][k];
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

export function ridgeFit(design, target, ridge) {
  // Ridge least-squares coefficients via the regularised normal equations. The
  // penalty is not applied to the constant term at index 0.
  const n = design[0].length;
  const ata = Array.from({ length: n }, () => new Array(n).fill(0));
  const atb = new Array(n).fill(0);
  for (let r = 0; r < design.length; r++) {
    const row = design[r];
    const value = target[r];
    for (let i = 0; i < n; i++) {
      atb[i] += row[i] * value;
      for (let j = 0; j < n; j++) ata[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < n; i++) ata[i][i] += ridge;
  return solve(ata, atb);
}

/**
 * P-spline building blocks for the GAM term figures: evenly spaced cubic B-splines (as in pyGAM), their periodic
 * version, difference penalties, Kronecker products and a Cholesky solver. Bases have at most a few hundred columns.
 */

export type Matrix = number[][]

/** The cardinal cubic B-spline on [0, 4), the building block of every evenly spaced cubic basis. */
export function cardinalCubic(u: number): number {
  if (u < 0 || u >= 4) return 0
  if (u < 1) return (u * u * u) / 6
  if (u < 2) return (-3 * u ** 3 + 12 * u * u - 12 * u + 4) / 6
  if (u < 3) return (3 * u ** 3 - 24 * u * u + 60 * u - 44) / 6
  return (4 - u) ** 3 / 6
}

/**
 * The k cubic B-splines on evenly spaced knots that cover [lo, hi] with k - 3 intervals. The knots continue past both
 * ends, so every basis function has full support and the basis sums to 1 on [lo, hi].
 */
export function psplineRow(x: number, lo: number, hi: number, k: number): number[] {
  const t = (x - lo) / ((hi - lo) / (k - 3))
  return Array.from({ length: k }, (_, j) => cardinalCubic(t - j + 3))
}

/** The k periodic cubic B-splines on [lo, hi): k knot intervals on a circle, so the basis wraps from hi back to lo. */
export function periodicRow(x: number, lo: number, hi: number, k: number): number[] {
  const t = (x - lo) / ((hi - lo) / k)
  return Array.from({ length: k }, (_, j) => cardinalCubic((((t - j + 2) % k) + k) % k))
}

/** The (k - order) × k matrix of order-th differences, so that (D β)_j = Δ^order β_j. */
export function diffMatrix(k: number, order: number): Matrix {
  let D: Matrix = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))
  for (let o = 0; o < order; o++) D = D.slice(1).map((row, i) => row.map((v, j) => v - D[i][j]))
  return D
}

/** The k × k circulant second-difference matrix: β_{j-1} - 2β_j + β_{j+1} with indices taken mod k. */
export function cyclicSecondDiff(k: number): Matrix {
  return Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (j === i ? -2 : j === (i + 1) % k || j === (i + k - 1) % k ? 1 : 0)),
  )
}

export const eye = (k: number): Matrix =>
  Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))

/** Dᵀ D. */
export function gram(D: Matrix): Matrix {
  const k = D[0].length
  const S: Matrix = Array.from({ length: k }, () => Array(k).fill(0))
  for (const row of D)
    for (let i = 0; i < k; i++) {
      if (row[i] === 0) continue
      for (let j = 0; j < k; j++) S[i][j] += row[i] * row[j]
    }
  return S
}

/** Kronecker product A ⊗ B. */
export function kron(A: Matrix, B: Matrix): Matrix {
  const [p, q] = [B.length, B[0].length]
  return Array.from({ length: A.length * p }, (_, r) =>
    Array.from({ length: A[0].length * q }, (_, c) => A[Math.floor(r / p)][Math.floor(c / q)] * B[r % p][c % q]),
  )
}

/** Row-wise Kronecker product of two basis rows: a ⊗ b, with the index of b running fastest. */
export function rowKron(a: number[], b: number[]): number[] {
  return a.flatMap((ai) => b.map((bj) => ai * bj))
}

/** Bᵀ W B for design rows B and optional weights w. Skips zeros, since B-spline rows are sparse. */
export function crossprod(B: Matrix, w?: number[]): Matrix {
  const p = B[0].length
  const M: Matrix = Array.from({ length: p }, () => Array(p).fill(0))
  B.forEach((row, i) => {
    const wi = w ? w[i] : 1
    const nz: number[] = []
    row.forEach((v, j) => v !== 0 && nz.push(j))
    for (const a of nz) for (const b of nz) M[a][b] += wi * row[a] * row[b]
  })
  return M
}

/** Bᵀ W y. */
export function crossprodY(B: Matrix, y: number[], w?: number[]): number[] {
  const out = Array(B[0].length).fill(0)
  B.forEach((row, i) => row.forEach((v, j) => v !== 0 && (out[j] += (w ? w[i] : 1) * v * y[i])))
  return out
}

/** A + Σ cᵢ Mᵢ. */
export function addScaled(A: Matrix, ...terms: [number, Matrix][]): Matrix {
  return A.map((row, i) => row.map((v, j) => terms.reduce((s, [c, M]) => s + c * M[i][j], v)))
}

/** Lower Cholesky factor of a symmetric positive-definite matrix. A tiny jitter guards against round-off. */
export function cholesky(A: Matrix): Matrix {
  const n = A.length
  const L: Matrix = Array.from({ length: n }, () => Array(n).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i][j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]
    }
  return L
}

/** Solve (L Lᵀ) u = v. */
export function cholSolve(L: Matrix, v: number[]): number[] {
  const n = v.length
  const z = Array(n).fill(0)
  for (let i = 0; i < n; i++) {
    let s = v[i]
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k]
    z[i] = s / L[i][i]
  }
  const u = Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < n; k++) s -= L[k][i] * u[k]
    u[i] = s / L[i][i]
  }
  return u
}

/** tr(A⁻¹ G) for A = L Lᵀ: the effective degrees of freedom when G = Bᵀ W B. */
export function traceSolve(L: Matrix, G: Matrix): number {
  let t = 0
  for (let k = 0; k < G.length; k++)
    t += cholSolve(
      L,
      G.map((row) => row[k]),
    )[k]
  return t
}

export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

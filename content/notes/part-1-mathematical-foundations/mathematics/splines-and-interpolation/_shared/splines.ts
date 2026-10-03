/**
 * Splines and interpolants for the splines-and-interpolation notes. Plain functions, no React, so the maths can be
 * checked against scipy.interpolate from node. Sizes are tiny (tens of points), so dense solves are enough.
 *
 * Two families:
 * - Function interpolants y = f(x) through sorted data: piecewise linear, the interpolating polynomial, cubic splines
 *   with four end conditions, PCHIP, Akima and the cubic smoothing spline. All but the polynomial are returned as a
 *   piecewise polynomial `PP` in the local power basis, as scipy's PPoly stores them.
 * - Parametric curves C(u) in the plane: B-splines, NURBS, Bézier curves and Catmull–Rom splines.
 */

export type Vec2 = [number, number]

// ---------------------------------------------------------------------------------------------------------------------
// Linear algebra

/** Solve A u = v by Gaussian elimination with partial pivoting. Neither input is modified. */
export function solve(A: number[][], v: number[]): number[] {
  const n = v.length
  const M = A.map((row, i) => [...row, v[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    const piv = M[c][c] || 1e-300
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const u = Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * u[k]
    u[r] = s / (M[r][r] || 1e-300)
  }
  return u
}

/**
 * Thomas algorithm for a tridiagonal system: sub-diagonal a (a[0] unused), diagonal b, super-diagonal c (c[n-1]
 * unused), right-hand side d. O(n) and stable for diagonally dominant systems such as the cubic spline equations.
 */
export function thomas(a: number[], b: number[], c: number[], d: number[]): number[] {
  const n = d.length
  const cp = Array<number>(n).fill(0)
  const dp = Array<number>(n).fill(0)
  cp[0] = c[0] / b[0]
  dp[0] = d[0] / b[0]
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1]
    cp[i] = c[i] / m
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m
  }
  const x = Array<number>(n).fill(0)
  x[n - 1] = dp[n - 1]
  for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1]
  return x
}

// ---------------------------------------------------------------------------------------------------------------------
// Piecewise polynomials

/** Piecewise polynomial: on [breaks[i], breaks[i+1]], f(x) = sum_k coefs[i][k] (x - breaks[i])^k. */
export type PP = { breaks: number[]; coefs: number[][] }

/** Index of the interval holding x; points outside use the end intervals (extrapolation). */
export function ppInterval(breaks: number[], x: number): number {
  let lo = 0
  let hi = breaks.length - 2
  if (x <= breaks[0]) return 0
  if (x >= breaks[hi]) return hi
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (breaks[mid] <= x) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** Value, first and second derivative of a PP at x. */
export function ppEval(pp: PP, x: number): [number, number, number] {
  const i = ppInterval(pp.breaks, x)
  const c = pp.coefs[i]
  const d = x - pp.breaks[i]
  let f = 0
  let f1 = 0
  let f2 = 0
  for (let k = c.length - 1; k >= 0; k--) {
    f2 = f2 * d + 2 * f1
    f1 = f1 * d + f
    f = f * d + c[k]
  }
  return [f, f1, f2]
}

/** Cubic Hermite pieces from values y and slopes m at sorted x. */
export function hermitePP(x: number[], y: number[], m: number[]): PP {
  const coefs: number[][] = []
  for (let i = 0; i < x.length - 1; i++) {
    const h = x[i + 1] - x[i]
    const delta = (y[i + 1] - y[i]) / h
    coefs.push([y[i], m[i], (3 * delta - 2 * m[i] - m[i + 1]) / h, (m[i] + m[i + 1] - 2 * delta) / (h * h)])
  }
  return { breaks: [...x], coefs }
}

/** Piecewise linear interpolant. */
export function linearPP(x: number[], y: number[]): PP {
  const coefs = x.slice(0, -1).map((xi, i) => [y[i], (y[i + 1] - y[i]) / (x[i + 1] - xi)])
  return { breaks: [...x], coefs }
}

/** Cubic pieces from values y and second derivatives M at the knots. */
export function cubicFromMoments(x: number[], y: number[], M: number[]): PP {
  const coefs: number[][] = []
  for (let i = 0; i < x.length - 1; i++) {
    const h = x[i + 1] - x[i]
    const b = (y[i + 1] - y[i]) / h - (h * (2 * M[i] + M[i + 1])) / 6
    coefs.push([y[i], b, M[i] / 2, (M[i + 1] - M[i]) / (6 * h)])
  }
  return { breaks: [...x], coefs }
}

export type EndCondition = 'natural' | 'clamped' | 'not-a-knot' | 'periodic'

/**
 * Interpolating cubic spline through (x_i, y_i), solved for the second derivatives M_i. Interior rows are
 * h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1} = 6 (delta_i - delta_{i-1}). `clamped` means zero end slopes,
 * as in scipy's CubicSpline; `periodic` uses y_0 in place of y_n.
 */
export function cubicSplinePP(x: number[], yIn: number[], bc: EndCondition): PP {
  const n = x.length - 1
  const y = [...yIn]
  if (bc === 'periodic') y[n] = y[0]
  const h = x.slice(0, -1).map((xi, i) => x[i + 1] - xi)
  const delta = h.map((hi, i) => (y[i + 1] - y[i]) / hi)
  if (n === 1) return linearPP(x, y)
  if (bc === 'natural') {
    // Interior unknowns only: a strictly diagonally dominant tridiagonal system, solved by the Thomas algorithm.
    const m = n - 1
    const a = Array.from({ length: m }, (_, k) => (k === 0 ? 0 : h[k]))
    const b = Array.from({ length: m }, (_, k) => 2 * (h[k] + h[k + 1]))
    const c = Array.from({ length: m }, (_, k) => (k === m - 1 ? 0 : h[k + 1]))
    const d = Array.from({ length: m }, (_, k) => 6 * (delta[k + 1] - delta[k]))
    return cubicFromMoments(x, y, [0, ...thomas(a, b, c, d), 0])
  }
  const size = bc === 'periodic' ? n : n + 1
  const A = Array.from({ length: size }, () => Array<number>(size).fill(0))
  const r = Array<number>(size).fill(0)
  const col = (j: number) => (bc === 'periodic' ? ((j % n) + n) % n : j)
  for (let i = 1; i < n; i++) {
    A[i][col(i - 1)] += h[i - 1]
    A[i][col(i)] += 2 * (h[i - 1] + h[i])
    A[i][col(i + 1)] += h[i]
    r[i] = 6 * (delta[i] - delta[i - 1])
  }
  if (bc === 'periodic') {
    // Row 0 joins the last interval to the first: slope and curvature continuity across x_n = x_0.
    A[0][col(n - 1)] += h[n - 1]
    A[0][0] += 2 * (h[n - 1] + h[0])
    A[0][col(1)] += h[0]
    r[0] = 6 * (delta[0] - delta[n - 1])
    const M = solve(A, r)
    return cubicFromMoments(x, y, [...M, M[0]])
  }
  if (bc === 'clamped') {
    A[0][0] = 2 * h[0]
    A[0][1] = h[0]
    r[0] = 6 * delta[0]
    A[n][n - 1] = h[n - 1]
    A[n][n] = 2 * h[n - 1]
    r[n] = -6 * delta[n - 1]
  } else if (n === 2) {
    // Three points: the two not-a-knot conditions coincide, and the spline is the interpolating parabola.
    A[0][0] = 1
    A[0][1] = -1
    A[2][1] = 1
    A[2][2] = -1
  } else {
    // Not-a-knot: the third derivative is continuous at x_1 and x_{n-1}, so the first two and last two pieces are
    // one cubic each.
    A[0][0] = h[1]
    A[0][1] = -(h[0] + h[1])
    A[0][2] = h[0]
    A[n][n - 2] = h[n - 1]
    A[n][n - 1] = -(h[n - 2] + h[n - 1])
    A[n][n] = h[n - 2]
  }
  return cubicFromMoments(x, y, solve(A, r))
}

/** Fritsch–Carlson slopes as scipy's PchipInterpolator computes them: weighted harmonic means, zero at extrema. */
export function pchipSlopes(x: number[], y: number[]): number[] {
  const n = x.length
  const h = x.slice(0, -1).map((xi, i) => x[i + 1] - xi)
  const m = h.map((hi, i) => (y[i + 1] - y[i]) / hi)
  if (n === 2) return [m[0], m[0]]
  const d = Array<number>(n).fill(0)
  for (let k = 1; k < n - 1; k++) {
    if (m[k - 1] === 0 || m[k] === 0 || Math.sign(m[k - 1]) !== Math.sign(m[k])) continue
    const w1 = 2 * h[k] + h[k - 1]
    const w2 = h[k] + 2 * h[k - 1]
    d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
  }
  const edge = (h0: number, h1: number, m0: number, m1: number) => {
    const e = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if (Math.sign(e) !== Math.sign(m0)) return 0
    if (Math.sign(m0) !== Math.sign(m1) && Math.abs(e) > 3 * Math.abs(m0)) return 3 * m0
    return e
  }
  d[0] = edge(h[0], h[1], m[0], m[1])
  d[n - 1] = edge(h[n - 2], h[n - 3] ?? h[n - 2], m[n - 2], m[n - 3] ?? m[n - 2])
  return d
}

export const pchipPP = (x: number[], y: number[]): PP => hermitePP(x, y, pchipSlopes(x, y))

/** Akima's slopes (scipy's Akima1DInterpolator, method 'akima'): a weighted average of the two neighbouring slopes. */
export function akimaSlopes(x: number[], y: number[]): number[] {
  const n = x.length
  if (n === 2) {
    const s = (y[1] - y[0]) / (x[1] - x[0])
    return [s, s]
  }
  const m = Array<number>(n + 3).fill(0)
  for (let i = 0; i < n - 1; i++) m[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
  m[1] = 2 * m[2] - m[3]
  m[0] = 2 * m[1] - m[2]
  m[n + 1] = 2 * m[n] - m[n - 1]
  m[n + 2] = 2 * m[n + 1] - m[n]
  const dm = m.slice(0, -1).map((mi, i) => Math.abs(m[i + 1] - mi))
  const f12 = Array.from({ length: n }, (_, i) => dm[i + 2] + dm[i])
  const cutoff = 1e-9 * Math.max(...f12)
  return Array.from({ length: n }, (_, i) => {
    if (!(f12[i] > cutoff)) return 0.5 * (m[i + 3] + m[i])
    return m[i + 1] + (dm[i] / f12[i]) * (m[i + 2] - m[i + 1])
  })
}

export const akimaPP = (x: number[], y: number[]): PP => hermitePP(x, y, akimaSlopes(x, y))

/**
 * Cubic smoothing spline minimising sum (y_i - f(x_i))^2 + lam * int f''^2 (Reinsch). With Q the n x (n-2) second
 * difference matrix and R the (n-2) x (n-2) tridiagonal Gram matrix, (R + lam Q'Q) gamma = Q'y gives the interior second
 * derivatives gamma and the fitted values g = y - lam Q gamma. Matches scipy's make_smoothing_spline(x, y, lam=lam).
 */
export function smoothingSplinePP(x: number[], y: number[], lam: number): PP {
  const n = x.length
  if (n < 3) return linearPP(x, y)
  const h = x.slice(0, -1).map((xi, i) => x[i + 1] - xi)
  const m = n - 2
  // Q[i][j] for data row i and interior knot column j (knot j + 1).
  const Q = Array.from({ length: n }, () => Array<number>(m).fill(0))
  for (let j = 0; j < m; j++) {
    Q[j][j] = 1 / h[j]
    Q[j + 1][j] = -1 / h[j] - 1 / h[j + 1]
    Q[j + 2][j] = 1 / h[j + 1]
  }
  const A = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let s = 0
      for (let k = 0; k < n; k++) s += Q[k][i] * Q[k][j]
      const R = i === j ? (h[i] + h[i + 1]) / 3 : Math.abs(i - j) === 1 ? h[Math.max(i, j)] / 6 : 0
      return R + lam * s
    }),
  )
  const rhs = Array.from({ length: m }, (_, j) => y.reduce((s, yk, k) => s + Q[k][j] * yk, 0))
  const gamma = solve(A, rhs)
  const g = y.map((yk, k) => yk - lam * Q[k].reduce((s, q, j) => s + q * gamma[j], 0))
  return cubicFromMoments(x, g, [0, ...gamma, 0])
}

// ---------------------------------------------------------------------------------------------------------------------
// Polynomial interpolation

/** Newton divided differences a_k = f[x_0, ..., x_k]. */
export function newtonCoefficients(x: number[], y: number[]): number[] {
  const a = [...y]
  for (let j = 1; j < x.length; j++)
    for (let i = x.length - 1; i >= j; i--) a[i] = (a[i] - a[i - 1]) / (x[i] - x[i - j])
  return a
}

/** Value, first and second derivative of the Newton form sum_k a_k prod_{j<k} (u - x_j), by nested multiplication. */
export function newtonEval(x: number[], a: number[], u: number): [number, number, number] {
  let p = a[a.length - 1]
  let dp = 0
  let d2p = 0
  for (let k = a.length - 2; k >= 0; k--) {
    const w = u - x[k]
    d2p = d2p * w + 2 * dp
    dp = dp * w + p
    p = p * w + a[k]
  }
  return [p, dp, d2p]
}

/** Barycentric weights w_j = 1 / prod_{k != j} (x_j - x_k). */
export function barycentricWeights(x: number[]): number[] {
  return x.map((xj, j) => 1 / x.reduce((s, xk, k) => (k === j ? s : s * (xj - xk)), 1))
}

/** The interpolating polynomial by the second (true) barycentric formula: stable for any nodes. */
export function barycentricEval(x: number[], y: number[], w: number[], u: number): number {
  let num = 0
  let den = 0
  for (let j = 0; j < x.length; j++) {
    const d = u - x[j]
    if (d === 0) return y[j]
    num += (w[j] / d) * y[j]
    den += w[j] / d
  }
  return num / den
}

/** Lebesgue function sum_j |l_j(u)|: the factor by which interpolation can amplify errors in the data at u. */
export function lebesgueFunction(x: number[], w: number[], u: number): number {
  let num = 0
  let den = 0
  for (let j = 0; j < x.length; j++) {
    const d = u - x[j]
    if (d === 0) return 1
    num += Math.abs(w[j] / d)
    den += w[j] / d
  }
  return num / Math.abs(den)
}

/** Node polynomial omega(u) = prod_j (u - x_j), which sets the size of the interpolation error. */
export const nodePolynomial = (x: number[], u: number) => x.reduce((s, xj) => s * (u - xj), 1)

/** n Chebyshev points of the first kind on [a, b]: the roots of T_n, mapped from [-1, 1]. */
export function chebyshevNodes(n: number, a: number, b: number): number[] {
  return Array.from({ length: n }, (_, k) => {
    const c = Math.cos(((2 * (n - 1 - k) + 1) * Math.PI) / (2 * n))
    return (a + b) / 2 + ((b - a) / 2) * c
  })
}

export const equispacedNodes = (n: number, a: number, b: number) =>
  Array.from({ length: n }, (_, k) => (n === 1 ? (a + b) / 2 : a + ((b - a) * k) / (n - 1)))

// ---------------------------------------------------------------------------------------------------------------------
// B-splines

/** Open-uniform (clamped) knots for nCtrl control points of degree p on [0, 1]: ends repeated p + 1 times. */
export function openUniformKnots(nCtrl: number, p: number): number[] {
  const inner = nCtrl - p - 1
  return [
    ...Array<number>(p + 1).fill(0),
    ...Array.from({ length: inner }, (_, j) => (j + 1) / (inner + 1)),
    ...Array<number>(p + 1).fill(1),
  ]
}

/** Uniform knots t_i = (i - p) / (nCtrl - p), so the curve's domain [t_p, t_nCtrl] is [0, 1]. */
export function uniformKnots(nCtrl: number, p: number): number[] {
  return Array.from({ length: nCtrl + p + 1 }, (_, i) => (i - p) / (nCtrl - p))
}

/** Parameter domain [t_p, t_{m-p}] of a B-spline of degree p on knots t (m + 1 knots). */
export const bsplineDomain = (t: number[], p: number): [number, number] => [t[p], t[t.length - 1 - p]]

/**
 * Values at u of all B-splines B_{i,p}, i = 0 .. m - p - 1, by the Cox–de Boor recursion. Degree 0 is the indicator
 * of [t_i, t_{i+1}); at the right end of the domain the last non-empty interval is closed, so the curve reaches its end.
 */
export function bsplineBasis(u: number, t: number[], p: number): number[] {
  const m = t.length - 1
  const hi = t[m - p]
  const at = Math.min(u, hi)
  let span = -1
  for (let i = 0; i < m; i++) {
    if (t[i] < t[i + 1] && ((at >= t[i] && at < t[i + 1]) || (at === hi && t[i + 1] === hi))) {
      span = i
      if (at < t[i + 1]) break
    }
  }
  let b: number[] = Array.from({ length: m }, (_, i) => (i === span ? 1 : 0))
  for (let d = 1; d <= p; d++) {
    const next = Array<number>(m - d).fill(0)
    for (let i = 0; i < m - d; i++) {
      const l = t[i + d] - t[i]
      const r = t[i + d + 1] - t[i + 1]
      next[i] = (l > 0 ? ((at - t[i]) / l) * b[i] : 0) + (r > 0 ? ((t[i + d + 1] - at) / r) * b[i + 1] : 0)
    }
    b = next
  }
  return b
}

/**
 * k-th derivative of every B-spline of degree p at u, from
 * B'_{i,p} = p B_{i,p-1} / (t_{i+p} - t_i) - p B_{i+1,p-1} / (t_{i+p+1} - t_{i+1}).
 */
export function bsplineBasisDerivative(u: number, t: number[], p: number, k: number): number[] {
  if (k === 0) return bsplineBasis(u, t, p)
  if (k > p) return Array<number>(t.length - 1 - p).fill(0)
  const lower = bsplineBasisDerivative(u, t, p - 1, k - 1)
  const count = t.length - 1 - p
  return Array.from({ length: count }, (_, i) => {
    const l = t[i + p] - t[i]
    const r = t[i + p + 1] - t[i + 1]
    return (l > 0 ? (p * lower[i]) / l : 0) - (r > 0 ? (p * lower[i + 1]) / r : 0)
  })
}

export type Evaluated = { p: Vec2; d1: Vec2; d2: Vec2 }

const combine = (w: number[], P: Vec2[]): Vec2 => [
  w.reduce((s, wi, i) => s + wi * P[i][0], 0),
  w.reduce((s, wi, i) => s + wi * P[i][1], 0),
]

/**
 * Point, first and second derivative of the NURBS curve sum_i N_i w_i P_i / sum_i N_i w_i (a B-spline curve when all
 * weights are 1), by the quotient rule on the homogeneous numerator A(u) and denominator W(u).
 */
export function nurbsEval(P: Vec2[], w: number[], t: number[], p: number, u: number): Evaluated {
  const N = [0, 1, 2].map((k) => bsplineBasisDerivative(u, t, p, k))
  const Wk = N.map((Nk) => Nk.reduce((s, v, i) => s + v * w[i], 0))
  const Ak = N.map((Nk) =>
    combine(
      Nk.map((v, i) => v * w[i]),
      P,
    ),
  )
  const C: Vec2 = [Ak[0][0] / Wk[0], Ak[0][1] / Wk[0]]
  const d1: Vec2 = [0, 1].map((j) => (Ak[1][j] - Wk[1] * C[j]) / Wk[0]) as Vec2
  const d2: Vec2 = [0, 1].map((j) => (Ak[2][j] - 2 * Wk[1] * d1[j] - Wk[2] * C[j]) / Wk[0]) as Vec2
  return { p: C, d1, d2 }
}

/** Rational basis R_i = N_i w_i / sum_j N_j w_j (the B-spline basis when all weights are 1). */
export function nurbsBasis(u: number, t: number[], p: number, w: number[]): number[] {
  const N = bsplineBasis(u, t, p)
  const W = N.reduce((s, v, i) => s + v * w[i], 0)
  return N.map((v, i) => (v * w[i]) / W)
}

/** Index k of the knot span [t_k, t_{k+1}) that holds u, clamped to the domain. */
export function knotSpan(t: number[], p: number, u: number): number {
  const [lo, hi] = bsplineDomain(t, p)
  const v = Math.min(Math.max(u, lo), hi)
  let k = p
  for (let i = p; i < t.length - 1 - p; i++) if (t[i] <= v && t[i] < t[i + 1]) k = i
  return k
}

/**
 * The triangle of de Boor's algorithm at u: level 0 holds the p + 1 control points that act on the span, and each
 * level blends neighbours with alpha = (u - t_i) / (t_{i+p+1-r} - t_i) until one point, C(u), remains. Weighted points
 * are blended in homogeneous coordinates and projected, so the same triangle serves NURBS.
 */
export function deBoorLevels(P: Vec2[], w: number[], t: number[], p: number, u: number): Vec2[][] {
  const k = knotSpan(t, p, u)
  let d = Array.from({ length: p + 1 }, (_, j) => {
    const i = j + k - p
    return [P[i][0] * w[i], P[i][1] * w[i], w[i]]
  })
  const project = (q: number[][]): Vec2[] => q.map((v) => [v[0] / v[2], v[1] / v[2]])
  const levels = [project(d)]
  for (let r = 1; r <= p; r++) {
    const next: number[][] = []
    for (let j = r; j <= p; j++) {
      const i = j + k - p
      const den = t[i + p + 1 - r] - t[i]
      const a = den > 0 ? (u - t[i]) / den : 0
      next.push(d[j - r].map((v, c) => (1 - a) * v + a * d[j - r + 1][c]))
    }
    d = next
    levels.push(project(d))
  }
  return levels
}

/**
 * Boehm's knot insertion: insert u once. The curve is unchanged; one control point is added, and p of the old ones
 * are replaced by points on the old control polygon, Q_i = (1 - a_i) P_{i-1} + a_i P_i with
 * a_i = (u - t_i) / (t_{i+p} - t_i) for k - p + 1 <= i <= k.
 */
export function insertKnot(P: Vec2[], w: number[], t: number[], p: number, u: number) {
  const k = knotSpan(t, p, u)
  const H = P.map((q, i) => [q[0] * w[i], q[1] * w[i], w[i]])
  const Q: number[][] = []
  for (let i = 0; i <= P.length; i++) {
    if (i <= k - p) Q.push(H[i])
    else if (i >= k + 1) Q.push(H[i - 1])
    else {
      const a = (u - t[i]) / (t[i + p] - t[i])
      Q.push(H[i].map((v, c) => (1 - a) * H[i - 1][c] + a * v))
    }
  }
  return {
    points: Q.map((q) => [q[0] / q[2], q[1] / q[2]] as Vec2),
    weights: Q.map((q) => q[2]),
    knots: [...t.slice(0, k + 1), u, ...t.slice(k + 1)],
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bézier curves

/** Bernstein polynomial b_{i,n}(s) = C(n, i) s^i (1 - s)^(n - i). */
export function bernstein(n: number, i: number, s: number): number {
  let c = 1
  for (let j = 1; j <= i; j++) c = (c * (n - i + j)) / j
  return c * s ** i * (1 - s) ** (n - i)
}

/** Every level of de Casteljau's algorithm at s: repeated linear interpolation of the control polygon. */
export function deCasteljauLevels(P: Vec2[], s: number): Vec2[][] {
  const levels: Vec2[][] = [P.map((q) => [...q] as Vec2)]
  while (levels[levels.length - 1].length > 1) {
    const prev = levels[levels.length - 1]
    levels.push(prev.slice(1).map((q, i) => [(1 - s) * prev[i][0] + s * q[0], (1 - s) * prev[i][1] + s * q[1]]))
  }
  return levels
}

/** Bézier point and derivatives: C' = n sum Delta P_i b_{i,n-1}, C'' = n (n - 1) sum Delta^2 P_i b_{i,n-2}. */
export function bezierEval(P: Vec2[], s: number): Evaluated {
  const n = P.length - 1
  const diff = (Q: Vec2[]): Vec2[] => Q.slice(1).map((q, i) => [q[0] - Q[i][0], q[1] - Q[i][1]])
  const at = (Q: Vec2[], scale: number): Vec2 => {
    if (Q.length === 0) return [0, 0]
    const levels = deCasteljauLevels(Q, s)
    const q = levels[levels.length - 1][0]
    return [scale * q[0], scale * q[1]]
  }
  const D1 = diff(P)
  return { p: at(P, 1), d1: at(D1, n), d2: at(diff(D1), n * (n - 1)) }
}

/** Degree elevation: the same curve with one more control point, Q_i = (i / (n+1)) P_{i-1} + (1 - i / (n+1)) P_i. */
export function elevateBezier(P: Vec2[]): Vec2[] {
  const n = P.length - 1
  return Array.from({ length: n + 2 }, (_, i) => {
    const a = i / (n + 1)
    const prev = P[i - 1] ?? P[0]
    const here = P[i] ?? P[n]
    return [a * prev[0] + (1 - a) * here[0], a * prev[1] + (1 - a) * here[1]]
  })
}

// ---------------------------------------------------------------------------------------------------------------------
// Catmull–Rom and cardinal splines

/**
 * Knot parameters for a Catmull–Rom spline: t_{i+1} = t_i + |P_{i+1} - P_i|^alpha, with alpha = 0 (uniform), 1/2
 * (centripetal) or 1 (chordal). The end points are extended by reflection, P_{-1} = 2 P_0 - P_1.
 */
export function catmullRomKnots(P: Vec2[], alpha: number): { knots: number[]; ext: Vec2[] } {
  const n = P.length
  const ext: Vec2[] = [
    [2 * P[0][0] - P[1][0], 2 * P[0][1] - P[1][1]],
    ...P,
    [2 * P[n - 1][0] - P[n - 2][0], 2 * P[n - 1][1] - P[n - 2][1]],
  ]
  const knots = [0]
  for (let i = 1; i < ext.length; i++) {
    const d = Math.hypot(ext[i][0] - ext[i - 1][0], ext[i][1] - ext[i - 1][1])
    knots.push(knots[i - 1] + Math.max(d ** alpha, 1e-9))
  }
  // Shift so that the curve's domain starts at 0 (at P_0, which is ext[1]).
  return { knots: knots.map((k) => k - knots[1]), ext }
}

/**
 * Catmull–Rom tangent at ext[i] with respect to the knot parameter: the derivative at t_i of the quadratic through
 * ext[i-1], ext[i], ext[i+1] (Barry–Goldman), scaled by 1 - tension.
 */
function catmullRomTangent(v: number[], t: number[], i: number, tension: number): number {
  const d0 = (v[i] - v[i - 1]) / (t[i] - t[i - 1])
  const d1 = (v[i + 1] - v[i]) / (t[i + 1] - t[i])
  const d = (v[i + 1] - v[i - 1]) / (t[i + 1] - t[i - 1])
  return (1 - tension) * (d0 - d + d1)
}

/**
 * Evaluate one coordinate (values v on the extended points) of a Catmull–Rom spline at parameter u in [0, t_last],
 * returning value, first and second derivative with respect to u.
 */
export function catmullRomScalar(v: number[], knots: number[], tension: number, u: number): [number, number, number] {
  const n = v.length - 2 // real points
  let seg = 0
  for (let s = 1; s < n - 1; s++) if (u >= knots[s + 1]) seg = s
  const i = seg + 1 // ext index of the segment's start point
  const m0 = catmullRomTangent(v, knots, i, tension)
  const m1 = catmullRomTangent(v, knots, i + 1, tension)
  const pp = hermitePP([knots[i], knots[i + 1]], [v[i], v[i + 1]], [m0, m1])
  return ppEval(pp, u)
}

export function catmullRomEval(ext: Vec2[], knots: number[], tension: number, u: number): Evaluated {
  const xs = catmullRomScalar(
    ext.map((q) => q[0]),
    knots,
    tension,
    u,
  )
  const ys = catmullRomScalar(
    ext.map((q) => q[1]),
    knots,
    tension,
    u,
  )
  return { p: [xs[0], ys[0]], d1: [xs[1], ys[1]], d2: [xs[2], ys[2]] }
}

// ---------------------------------------------------------------------------------------------------------------------
// Differential geometry

/** Signed curvature (x'y'' - y'x'') / |C'|^3 of a parametric curve. Positive means turning left. */
export function curvature(d1: Vec2, d2: Vec2): number {
  const speed = Math.hypot(d1[0], d1[1])
  if (speed < 1e-12) return 0
  return (d1[0] * d2[1] - d1[1] * d2[0]) / speed ** 3
}

/** Unit left normal of a tangent vector. */
export function unitNormal(d1: Vec2): Vec2 {
  const s = Math.hypot(d1[0], d1[1]) || 1
  return [-d1[1] / s, d1[0] / s]
}

// ---------------------------------------------------------------------------------------------------------------------
// Curve models for the playground: one interface over every spline type

export type SplineType =
  'linear' | 'polynomial' | 'cubic' | 'pchip' | 'akima' | 'smoothing' | 'catmull-rom' | 'bspline' | 'bezier' | 'nurbs'

/** Types that interpolate or smooth data y = f(x); the rest are parametric curves C(u) in the plane. */
export const FUNCTION_TYPES: readonly SplineType[] = ['linear', 'polynomial', 'cubic', 'pchip', 'akima', 'smoothing']
export const isFunctionType = (t: SplineType) => FUNCTION_TYPES.includes(t)

export type CurveOptions = {
  type: SplineType
  points: Vec2[]
  weights: number[]
  end: EndCondition
  /** Catmull–Rom exponent: 0 uniform, 0.5 centripetal, 1 chordal. */
  alpha: number
  tension: number
  degree: number
  knots: number[]
  lambda: number
}

export type CurveModel = {
  domain: [number, number]
  at: (u: number) => Evaluated
  /** Parameter values of the joins between pieces (knots or breakpoints), interior only. */
  joins: number[]
  /** Order of continuity at the joins, e.g. 2 for C², Infinity for a single polynomial. */
  continuity: number
  /** Number of basis functions, or null for schemes that are not linear in the data. */
  basisCount: number | null
  /** All basis functions at u, one per point in `order`; null for non-linear schemes. */
  basis: ((u: number) => number[]) | null
  /** Levels of de Casteljau's or de Boor's algorithm at u, for curves that have them. */
  construction: ((u: number) => Vec2[][]) | null
  /** Point indices in the order the scheme uses them (sorted by x for function types). */
  order: number[]
  /** True if the curve passes through every point. */
  interpolates: boolean
}

/** Sorted, strictly increasing x (nudging ties apart) and the original indices in that order. */
export function sortedData(points: Vec2[]) {
  const order = points.map((_, i) => i).sort((a, b) => points[a][0] - points[b][0])
  const x: number[] = []
  const y: number[] = []
  order.forEach((i, k) => {
    x.push(k === 0 ? points[i][0] : Math.max(points[i][0], x[k - 1] + 1e-3))
    y.push(points[i][1])
  })
  return { order, x, y }
}

const unit = (n: number, j: number) => Array.from({ length: n }, (_, i) => (i === j ? 1 : 0))

function functionModel(o: CurveOptions): CurveModel {
  const { order, x, y } = sortedData(o.points)
  const n = x.length
  if (o.type === 'cubic' && o.end === 'periodic') y[n - 1] = y[0]
  const fit = (v: number[]): ((u: number) => [number, number, number]) => {
    if (o.type === 'polynomial') {
      const a = newtonCoefficients(x, v)
      return (u) => newtonEval(x, a, u)
    }
    const pp =
      o.type === 'linear'
        ? linearPP(x, v)
        : o.type === 'cubic'
          ? cubicSplinePP(x, v, o.end)
          : o.type === 'pchip'
            ? pchipPP(x, v)
            : o.type === 'akima'
              ? akimaPP(x, v)
              : smoothingSplinePP(x, v, o.lambda)
    return (u) => ppEval(pp, u)
  }
  const f = fit(y)
  const linear = o.type !== 'pchip' && o.type !== 'akima'
  let basis: CurveModel['basis'] = null
  if (linear) {
    if (o.type === 'polynomial') {
      const w = barycentricWeights(x)
      basis = (u) => Array.from({ length: n }, (_, j) => barycentricEval(x, unit(n, j), w, u))
    } else {
      const cardinals = Array.from({ length: n }, (_, j) => fit(unit(n, j)))
      basis = (u) => cardinals.map((c) => c(u)[0])
    }
  }
  const continuity = { linear: 0, polynomial: Infinity, cubic: 2, pchip: 1, akima: 1, smoothing: 2 }[o.type as 'linear']
  return {
    domain: [x[0], x[n - 1]],
    at: (u) => {
      const [v, d1, d2] = f(u)
      return { p: [u, v], d1: [1, d1], d2: [0, d2] }
    },
    joins: o.type === 'polynomial' ? [] : x.slice(1, -1),
    continuity,
    basisCount: linear ? n : null,
    basis,
    construction: null,
    order,
    interpolates: o.type !== 'smoothing',
  }
}

/** Distinct interior knots of a B-spline with their multiplicities. */
export function interiorKnots(t: number[], p: number): { value: number; multiplicity: number }[] {
  const [lo, hi] = bsplineDomain(t, p)
  const out: { value: number; multiplicity: number }[] = []
  for (const v of t) {
    if (v <= lo || v >= hi) continue
    const last = out[out.length - 1]
    if (last && last.value === v) last.multiplicity++
    else out.push({ value: v, multiplicity: 1 })
  }
  return out
}

export function buildCurve(o: CurveOptions): CurveModel {
  if (isFunctionType(o.type)) return functionModel(o)
  const P = o.points
  const order = P.map((_, i) => i)
  if (o.type === 'bezier') {
    const n = P.length - 1
    return {
      domain: [0, 1],
      at: (u) => bezierEval(P, u),
      joins: [],
      continuity: Infinity,
      basisCount: n + 1,
      basis: (u) => Array.from({ length: n + 1 }, (_, i) => bernstein(n, i, u)),
      construction: (u) => deCasteljauLevels(P, u),
      order,
      interpolates: false,
    }
  }
  if (o.type === 'catmull-rom') {
    const { knots, ext } = catmullRomKnots(P, o.alpha)
    const n = P.length
    const reflect = (v: number[]) => [2 * v[0] - v[1], ...v, 2 * v[n - 1] - v[n - 2]]
    return {
      domain: [0, knots[n]],
      at: (u) => catmullRomEval(ext, knots, o.tension, u),
      joins: knots.slice(2, n),
      continuity: 1,
      basisCount: n,
      basis: (u) => Array.from({ length: n }, (_, j) => catmullRomScalar(reflect(unit(n, j)), knots, o.tension, u)[0]),
      construction: null,
      order,
      interpolates: true,
    }
  }
  const w = o.type === 'nurbs' ? o.weights : P.map(() => 1)
  const p = o.degree
  const t = o.knots
  const inner = interiorKnots(t, p)
  return {
    domain: bsplineDomain(t, p),
    at: (u) => nurbsEval(P, w, t, p, u),
    joins: inner.map((k) => k.value),
    continuity: Math.min(p - 1, ...inner.map((k) => p - k.multiplicity)),
    basisCount: P.length,
    basis: (u) => nurbsBasis(u, t, p, w),
    construction: (u) => deBoorLevels(P, w, t, p, u),
    order,
    interpolates: false,
  }
}

export type CurveSample = { u: number; e: Evaluated; kappa: number }

/** Sample a model at `count` parameter values, with signed curvature. */
export function sampleCurve(model: CurveModel, count = 400): CurveSample[] {
  const [lo, hi] = model.domain
  return Array.from({ length: count }, (_, i) => {
    const u = lo + ((hi - lo) * i) / (count - 1)
    const e = model.at(u)
    return { u, e, kappa: curvature(e.d1, e.d2) }
  })
}

/** Bending energy int kappa^2 ds by the trapezoid rule in the parameter, with ds = |C'(u)| du. */
export function bendingEnergy(samples: CurveSample[]): number {
  let s = 0
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1]
    const b = samples[i]
    const fa = a.kappa ** 2 * Math.hypot(...a.e.d1)
    const fb = b.kappa ** 2 * Math.hypot(...b.e.d1)
    s += ((fa + fb) / 2) * (b.u - a.u)
  }
  return s
}

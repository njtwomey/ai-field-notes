import { normal, stream } from 'aifn-compute/foundation/random'
/**
 * Linear-Gaussian state-space model z_t = A z_{t−1} + w_t, x_t = C z_t + v_t, with w_t ~ N(0, Q) and v_t ~ N(0, R),
 * plus the Kalman filter and the Rauch–Tung–Striebel smoother. Plain arrays keep the code close to the equations;
 * the models here have at most four state dimensions.
 */

export type Vec = number[]
export type Mat = number[][]

export const matmul = (a: Mat, b: Mat): Mat =>
  a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
export const matvec = (a: Mat, x: Vec): Vec => a.map((row) => row.reduce((s, v, k) => s + v * x[k], 0))
export const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map((row) => row[j]))
export const add = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v + b[i][j]))
export const sub = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v - b[i][j]))
export const vadd = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i])
export const vsub = (a: Vec, b: Vec): Vec => a.map((v, i) => v - b[i])
export const identity = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)))
export const scale = (a: Mat, s: number): Mat => a.map((row) => row.map((v) => v * s))

/** Inverse by Gauss–Jordan elimination with partial pivoting. Adequate for the small, well-conditioned matrices here. */
export function inverse(a: Mat): Mat {
  const n = a.length
  const m = a.map((row, i) => [...row, ...identity(n)[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    ;[m[c], m[p]] = [m[p], m[c]]
    const pivot = m[c][c]
    for (let j = 0; j < 2 * n; j++) m[c][j] /= pivot
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = m[r][c]
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j]
    }
  }
  return m.map((row) => row.slice(n))
}

/** Lower Cholesky factor of a symmetric positive definite matrix, for sampling N(0, S) as L ε. */
export function cholesky(s: Mat): Mat {
  const n = s.length
  const l: Mat = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let v = s[i][j]
      for (let k = 0; k < j; k++) v -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(v, 0)) : v / l[j][j]
    }
  return l
}

export type Model = { A: Mat; C: Mat; Q: Mat; R: Mat }

export type FilterStep = {
  /** Predicted mean and covariance, μ_{t|t−1} and P_{t|t−1}. */
  mPred: Vec
  PPred: Mat
  /** Filtered mean and covariance, μ_{t|t} and P_{t|t}. */
  m: Vec
  P: Mat
  K: Mat
}

/**
 * Kalman filter. Predict: μ⁻ = A μ, P⁻ = A P Aᵀ + Q. Update: S = C P⁻ Cᵀ + R, K = P⁻ Cᵀ S⁻¹, μ = μ⁻ + K (x − C μ⁻),
 * and the covariance in Joseph form (I − KC) P⁻ (I − KC)ᵀ + K R Kᵀ, which stays symmetric positive definite in finite
 * precision.
 */
export function kalmanFilter(model: Model, xs: Vec[], m0: Vec, P0: Mat): FilterStep[] {
  const { A, C, Q, R } = model
  const At = transpose(A)
  const Ct = transpose(C)
  const I = identity(A.length)
  let m = m0
  let P = P0
  return xs.map((x) => {
    const mPred = matvec(A, m)
    const PPred = add(matmul(matmul(A, P), At), Q)
    const S = add(matmul(matmul(C, PPred), Ct), R)
    const K = matmul(matmul(PPred, Ct), inverse(S))
    m = vadd(mPred, matvec(K, vsub(x, matvec(C, mPred))))
    const J = sub(I, matmul(K, C))
    P = add(matmul(matmul(J, PPred), transpose(J)), matmul(matmul(K, R), transpose(K)))
    return { mPred, PPred, m, P, K }
  })
}

/**
 * Rauch–Tung–Striebel smoother. Backwards from the last filtered estimate: G = P_{t|t} Aᵀ P_{t+1|t}⁻¹,
 * μ_{t|T} = μ_{t|t} + G (μ_{t+1|T} − μ_{t+1|t}), P_{t|T} = P_{t|t} + G (P_{t+1|T} − P_{t+1|t}) Gᵀ.
 */
export function rtsSmoother(model: Model, steps: FilterStep[]): { m: Vec; P: Mat }[] {
  const At = transpose(model.A)
  const out: { m: Vec; P: Mat }[] = new Array(steps.length)
  const last = steps[steps.length - 1]
  out[steps.length - 1] = { m: last.m, P: last.P }
  for (let t = steps.length - 2; t >= 0; t--) {
    const { m, P } = steps[t]
    const next = steps[t + 1]
    const G = matmul(matmul(P, At), inverse(next.PPred))
    const ms = vadd(m, matvec(G, vsub(out[t + 1].m, next.mPred)))
    const Ps = add(P, matmul(matmul(G, sub(out[t + 1].P, next.PPred)), transpose(G)))
    out[t] = { m: ms, P: Ps }
  }
  return out
}

/**
 * Points on the ellipse {μ + L u : ‖u‖ = k} for a 2 × 2 covariance with Cholesky factor L: the contour at Mahalanobis
 * distance k.
 */
export function ellipse(mean: Vec, cov: Mat, k: number, points = 48): { x: number[]; y: number[] } {
  const L = cholesky(cov)
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= points; i++) {
    const a = (2 * Math.PI * i) / points
    const u = [k * Math.cos(a), k * Math.sin(a)]
    x.push(mean[0] + L[0][0] * u[0])
    y.push(mean[1] + L[1][0] * u[0] + L[1][1] * u[1])
  }
  return { x, y }
}

/** The 2 × 2 position block of a constant-velocity covariance (state order: x, y, vx, vy). */
export const positionBlock = (P: Mat): Mat => [
  [P[0][0], P[0][1]],
  [P[1][0], P[1][1]],
]

export type NoiseShape = 'round' | 'elongated'

/**
 * Constant-velocity motion in the plane with time step 1. State (x, y, vx, vy); the sensor reads the position. Q is the
 * discretised white-noise acceleration of intensity q. R is r I (round) or r U diag(4, 1/4) Uᵀ with U a 45° rotation
 * (elongated), which has the same determinant.
 */
export function constantVelocity(q: number, r: number, shape: NoiseShape): Model {
  const A = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]
  const C = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
  ]
  const Q = scale(
    [
      [1 / 3, 0, 1 / 2, 0],
      [0, 1 / 3, 0, 1 / 2],
      [1 / 2, 0, 1, 0],
      [0, 1 / 2, 0, 1],
    ],
    q,
  )
  // U diag(4, 1/4) Uᵀ for a 45° rotation U: diagonal (4 + 1/4)/2, off-diagonal (4 − 1/4)/2.
  const R =
    shape === 'round'
      ? scale(identity(2), r)
      : scale(
          [
            [2.125, 1.875],
            [1.875, 2.125],
          ],
          r,
        )
  return { A, C, Q, R }
}

/** Simulate a trajectory and its measurements from the model, starting at the origin with velocity (0.5, 0.2). */
export function simulate(model: Model, steps: number, seed: number): { z: Vec[]; x: Vec[] } {
  const g = stream(seed)
  const Lq = cholesky(model.Q)
  const Lr = cholesky(model.R)
  const noise = (L: Mat) =>
    matvec(
      L,
      L.map(() => normal(g)),
    )
  let state = [0, 0, 0.5, 0.2]
  const z: Vec[] = []
  const x: Vec[] = []
  for (let t = 0; t < steps; t++) {
    state = vadd(matvec(model.A, state), noise(Lq))
    z.push(state)
    x.push(vadd(matvec(model.C, state), noise(Lr)))
  }
  return { z, x }
}

/** Root-mean-square Euclidean distance between estimated and true positions. */
export const positionRmse = (est: Vec[], truth: Vec[]) =>
  Math.sqrt(est.reduce((s, e, t) => s + (e[0] - truth[t][0]) ** 2 + (e[1] - truth[t][1]) ** 2, 0) / est.length)

/** Iterate the Riccati recursion to convergence and return the steady-state gain. */
export function steadyStateGain(model: Model): Mat {
  const n = model.A.length
  const xs = Array.from({ length: 400 }, () => new Array<number>(model.C.length).fill(0))
  const steps = kalmanFilter(model, xs, new Array<number>(n).fill(0), scale(identity(n), 10))
  return steps[steps.length - 1].K
}

/** Eigenvalues of a 2 × 2 matrix with real eigenvalues, largest first. */
export function eigenvalues2(m: Mat): [number, number] {
  const tr = m[0][0] + m[1][1]
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
  const d = Math.sqrt(Math.max((tr * tr) / 4 - det, 0))
  return [tr / 2 + d, tr / 2 - d]
}

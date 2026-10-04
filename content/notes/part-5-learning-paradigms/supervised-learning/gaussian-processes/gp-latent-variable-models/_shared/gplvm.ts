/**
 * A small Gaussian process latent variable model for the figures in this category: squared exponential kernel,
 * a standard normal prior on the latent points, optional back constraints, and Adam on the log posterior. Matrices
 * are at most a few dozen rows, so dense loops are enough; the data may have thousands of columns, which enter only
 * through YYᵀ and the predictive mean.
 */
import { cholesky, cholSolve, logDet, type Matrix } from '../../_shared/gp'
import { normal, stream } from 'aifn/foundation/random'

export type Point = [number, number]

export type GplvmState = {
  /** Latent points, one row per data point, Q = 2 columns. */
  X: Point[]
  ell: number
  sf: number
  sn: number
  /** ln p(Y | X, θ) + ln p(X). */
  logPost: number
}

export type GplvmFit = {
  snapshots: GplvmState[]
  /** Iteration number of each snapshot. */
  iterations: number[]
}

export type FitOptions = {
  iterations: number
  snapshotEvery: number
  learningRate?: number
  /** Inverse width γ of an RBF back constraint X = K_bc A, with K_bc = exp(−γ‖yₙ − yₘ‖²/2). Absent: free X. */
  backConstraint?: number
  /** With back constraints: use X0 as the initial A directly, rather than solving K_bc A = X0. */
  initialAIsX0?: boolean
}

const sqDist = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0)

/** YYᵀ. Plain loops over the symmetric half: rows can be thousands of entries long (the font manifold). */
export function outer(Y: Matrix): Matrix {
  const n = Y.length
  const out: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    const a = Y[i]
    for (let j = 0; j <= i; j++) {
      const b = Y[j]
      let s = 0
      for (let d = 0; d < a.length; d++) s += a[d] * b[d]
      out[i][j] = s
      out[j][i] = s
    }
  }
  return out
}

/** A⁻¹ from the Cholesky factor of A. */
function inverse(l: Matrix): Matrix {
  const n = l.length
  const cols = Array.from({ length: n }, (_, j) =>
    cholSolve(
      l,
      Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)),
    ),
  )
  return cols[0].map((_, i) => cols.map((c) => c[i]))
}

function matMul(a: Matrix, b: Matrix): Matrix {
  const m = b[0].length
  return a.map((row) => {
    const out = new Array<number>(m).fill(0)
    for (let k = 0; k < row.length; k++) {
      const v = row[k]
      const bk = b[k]
      for (let j = 0; j < m; j++) out[j] += v * bk[j]
    }
    return out
  })
}

/** Eigenvalues (descending) and eigenvectors (columns) of a symmetric matrix, by cyclic Jacobi rotations. */
export function eigSymmetric(a: Matrix): { values: number[]; vectors: Matrix } {
  const n = a.length
  const m = a.map((r) => [...r])
  const v: Matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p][q] ** 2
    if (off < 1e-20) break
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-300) continue
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p]
          const mkq = m[k][q]
          m[k][p] = c * mkp - s * mkq
          m[k][q] = s * mkp + c * mkq
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k]
          const mqk = m[q][k]
          m[p][k] = c * mpk - s * mqk
          m[q][k] = s * mpk + c * mqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p]
          const vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
    }
  }
  const order = m.map((r, i) => [r[i], i] as const).sort((x, y) => y[0] - x[0])
  return { values: order.map(([val]) => val), vectors: v.map((row) => order.map(([, i]) => row[i])) }
}

/** Centres the columns of Y. */
export function centre(Y: Matrix): { Y: Matrix; mean: number[] } {
  const mean = Y[0].map((_, j) => Y.reduce((s, r) => s + r[j], 0) / Y.length)
  return { Y: Y.map((r) => r.map((v, j) => v - mean[j])), mean }
}

/** The first two principal component scores of centred Y, scaled so that the first has unit standard deviation. */
export function pcaScores(Y: Matrix): Point[] {
  const { values, vectors } = eigSymmetric(outer(Y))
  const scores = vectors.map((r): Point => [r[0] * Math.sqrt(values[0]), r[1] * Math.sqrt(Math.max(values[1], 0))])
  const sd = Math.sqrt(scores.reduce((s, p) => s + p[0] * p[0], 0) / scores.length)
  return scores.map(([a, b]) => [a / sd, b / sd])
}

export function randomLatent(n: number, seed: number, scale = 0.3): Point[] {
  const g = stream(seed)
  return Array.from({ length: n }, (): Point => [scale * normal(g), scale * normal(g)])
}

/** Log posterior and its gradients in X and in the log hyperparameters. */
function objective(X: Point[], logEll: number, logSf: number, logSn: number, YYt: Matrix, D: number) {
  const n = X.length
  const ell = Math.exp(logEll)
  const sf2 = Math.exp(2 * logSf)
  const sn2 = Math.exp(2 * logSn)
  const d2 = X.map((a) => X.map((b) => sqDist(a, b)))
  const Kf = d2.map((r) => r.map((v) => sf2 * Math.exp(-v / (2 * ell * ell))))
  const l = cholesky(Kf.map((r, i) => r.map((v, j) => (i === j ? v + sn2 : v))))
  const Ki = inverse(l)
  const KiYYt = matMul(Ki, YYt)
  const trace = KiYYt.reduce((s, r, i) => s + r[i], 0)
  const prior = X.reduce((s, p) => s + p[0] * p[0] + p[1] * p[1], 0)
  const logPost = -0.5 * D * n * Math.log(2 * Math.PI) - 0.5 * D * logDet(l) - 0.5 * trace - 0.5 * prior
  // G = ∂ln p(Y|X)/∂K = ½(K⁻¹YYᵀK⁻¹ − D K⁻¹).
  const A = matMul(KiYYt, Ki)
  const G = A.map((r, i) => r.map((v, j) => 0.5 * (v - D * Ki[i][j])))
  const W = G.map((r, i) => r.map((v, j) => v * Kf[i][j]))
  const gX = X.map((xi, i): Point => {
    let g0 = 0
    let g1 = 0
    for (let m = 0; m < n; m++) {
      g0 += W[i][m] * (xi[0] - X[m][0])
      g1 += W[i][m] * (xi[1] - X[m][1])
    }
    return [(-2 / (ell * ell)) * g0 - xi[0], (-2 / (ell * ell)) * g1 - xi[1]]
  })
  let gEll = 0
  let gSf = 0
  let gSn = 0
  for (let i = 0; i < n; i++) {
    gSn += 2 * sn2 * G[i][i]
    for (let j = 0; j < n; j++) {
      gEll += (W[i][j] * d2[i][j]) / (ell * ell)
      gSf += 2 * W[i][j]
    }
  }
  return { logPost, gX, gEll, gSf, gSn }
}

/** Settings of an incremental fit: everything in FitOptions except the schedule. */
export type FitterOptions = Omit<FitOptions, 'iterations' | 'snapshotEvery'>

export type GplvmFitter = {
  /** Adam steps taken so far. */
  readonly iteration: number
  /** Takes n more Adam steps (n = 0 is allowed) and returns the state reached. */
  run: (n: number) => GplvmState
}

/**
 * A GP-LVM fit with Q = 2 by Adam ascent on the log posterior, advanced a few steps at a time. A widget can run it in
 * chunks between frames, so that a fit on large data never blocks the page.
 */
export function gplvmFitter(Y: Matrix, X0: Point[], opts: FitterOptions = {}): GplvmFitter {
  const { learningRate = 0.05, backConstraint, initialAIsX0 = false } = opts
  const n = Y.length
  const D = Y[0].length
  // The likelihood sees the data only through the n × n matrix YYᵀ, so each step costs O(n³) whatever D is.
  const YYt = outer(Y)
  // With back constraints the free parameters are A, and X = K_bc A. A starts where K_bc A reproduces X0.
  const Kbc = backConstraint ? Y.map((a) => Y.map((b) => Math.exp((-backConstraint / 2) * sqDist(a, b)))) : undefined
  let P: Point[]
  if (Kbc && !initialAIsX0) {
    const l = cholesky(Kbc.map((r, i) => r.map((v, j) => (i === j ? v + 1e-3 : v))))
    const c0 = cholSolve(
      l,
      X0.map((p) => p[0]),
    )
    const c1 = cholSolve(
      l,
      X0.map((p) => p[1]),
    )
    P = c0.map((v, i): Point => [v, c1[i]])
  } else {
    P = X0.map((p): Point => [p[0], p[1]])
  }
  const latent = (): Point[] =>
    Kbc
      ? Kbc.map((r): Point => [r.reduce((s, v, m) => s + v * P[m][0], 0), r.reduce((s, v, m) => s + v * P[m][1], 0)])
      : P
  let h = [0, 0, Math.log(0.1)]
  const size = 2 * n + 3
  const m1 = new Array<number>(size).fill(0)
  const m2 = new Array<number>(size).fill(0)
  let t = 0
  const evaluate = () => {
    const X = latent()
    return { X, o: objective(X, h[0], h[1], h[2], YYt, D) }
  }
  const update = (o: ReturnType<typeof objective>) => {
    const gP = Kbc
      ? Kbc.map((r): Point => [
          r.reduce((s, v, m) => s + v * o.gX[m][0], 0),
          r.reduce((s, v, m) => s + v * o.gX[m][1], 0),
        ])
      : o.gX
    const grad = [...gP.flat(), o.gEll, o.gSf, o.gSn]
    const params = [...P.flat(), ...h]
    const step = t + 1
    for (let k = 0; k < size; k++) {
      m1[k] = 0.9 * m1[k] + 0.1 * grad[k]
      m2[k] = 0.999 * m2[k] + 0.001 * grad[k] * grad[k]
      const mh = m1[k] / (1 - 0.9 ** step)
      const vh = m2[k] / (1 - 0.999 ** step)
      params[k] += (learningRate * mh) / (Math.sqrt(vh) + 1e-8)
    }
    P = Array.from({ length: n }, (_, i): Point => [params[2 * i], params[2 * i + 1]])
    h = params.slice(2 * n)
    t++
  }
  let current = evaluate()
  return {
    get iteration() {
      return t
    },
    run: (steps) => {
      for (let k = 0; k < steps; k++) {
        update(current.o)
        current = evaluate()
      }
      return {
        X: current.X.map((p): Point => [p[0], p[1]]),
        ell: Math.exp(h[0]),
        sf: Math.exp(h[1]),
        sn: Math.exp(h[2]),
        logPost: current.o.logPost,
      }
    },
  }
}

/** Fits a GP-LVM with Q = 2 by Adam ascent on the log posterior, recording a snapshot every `snapshotEvery` steps. */
export function fitGplvm(Y: Matrix, X0: Point[], opts: FitOptions): GplvmFit {
  const { iterations, snapshotEvery, ...rest } = opts
  const fitter = gplvmFitter(Y, X0, rest)
  const snapshots: GplvmState[] = [fitter.run(0)]
  const its: number[] = [0]
  for (let t = snapshotEvery; t <= iterations; t += snapshotEvery) {
    snapshots.push(fitter.run(snapshotEvery))
    its.push(t)
  }
  return { snapshots, iterations: its }
}

export type Predictor = {
  /** Posterior mean of the (centred) data vector at latent point x. */
  mean: (x: Point) => number[]
  /** Posterior standard deviation of each output of f at x, noise excluded. The same for every output. */
  sd: (x: Point) => number
  /** Posterior mean of the chosen outputs only: O(n) per output rather than O(nD) for the whole vector. */
  meanAt: (x: Point, dims: number[]) => number[]
  /** Gradient of output d's posterior mean with respect to x. */
  meanGrad: (x: Point, d: number) => Point
}

/** The GP posterior over the mapping, given the fitted latent points and hyperparameters. */
export function predictor(Y: Matrix, s: GplvmState): Predictor {
  const { X, ell, sf, sn } = s
  const n = X.length
  const sf2 = sf * sf
  const K = X.map((a) => X.map((b) => sf2 * Math.exp(-sqDist(a, b) / (2 * ell * ell))))
  const l = cholesky(K.map((r, i) => r.map((v, j) => (i === j ? v + sn * sn : v))))
  const Ki = inverse(l)
  const alpha = matMul(Ki, Y)
  const kvec = (x: Point) => X.map((xm) => sf2 * Math.exp(-sqDist(x, xm) / (2 * ell * ell)))
  return {
    mean: (x) => {
      const k = kvec(x)
      return alpha[0].map((_, d) => k.reduce((acc, km, m) => acc + km * alpha[m][d], 0))
    },
    meanAt: (x, dims) => {
      const k = kvec(x)
      return dims.map((d) => k.reduce((acc, km, m) => acc + km * alpha[m][d], 0))
    },
    sd: (x) => {
      const k = kvec(x)
      let q = 0
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q += k[i] * Ki[i][j] * k[j]
      return Math.sqrt(Math.max(sf2 - q, 0))
    },
    meanGrad: (x, d) => {
      const k = kvec(x)
      let g0 = 0
      let g1 = 0
      for (let m = 0; m < n; m++) {
        const w = (k[m] * alpha[m][d]) / (ell * ell)
        g0 -= w * (x[0] - X[m][0])
        g1 -= w * (x[1] - X[m][1])
      }
      return [g0, g1]
    },
  }
}

/**
 * Linear-Gaussian state-space models z_t = A z_{t−1} + w_t, y_t = C z_t + v_t, w_t ~ N(0, Q), v_t ~ N(0, R), with
 * z₀ ~ N(m₀, P₀): simulation, the Kalman filter, the Rauch–Tung–Striebel smoother, the steady-state filter and EM.
 *
 * Convention: (m₀, P₀) describes z₀, which is not observed; the first observation y₁ is of z₁ = A z₀ + w₁, so the
 * filter predicts before its first update. Rows of y that contain NaN are missing: the filter predicts through them.
 */

import { child, normals, type Stream } from 'aifn/foundation/random'
import { toFlat, type Matrix, type Tensor } from 'aifn/foundation/tensor'
import {
  addM,
  addV,
  eyeM,
  matmul,
  matT,
  matvec,
  sandwich,
  scaleM,
  solveM,
  sqrtPsd,
  stackMats,
  stackVecs,
  subM,
  subV,
  symmetrise,
  toMat,
  toSeries,
  toVec,
  transpose,
  vecT,
  zerosM,
  type Mat,
  type MatrixLike,
  type Vec,
  type VectorLike,
} from './mat'

/** A linear-Gaussian state-space model. Scalars stand for 1×1 matrices (and a length-1 m₀). */
export type StateSpaceModel = {
  /** Transition A (n×n). */
  A: MatrixLike | number
  /** Observation C (m×n). */
  C: MatrixLike | number
  /** Process-noise covariance Q (n×n, positive semi-definite; zero rows are allowed). */
  Q: MatrixLike | number
  /** Observation-noise covariance R (m×m, positive semi-definite; zero rows are allowed). */
  R: MatrixLike | number
  /** Mean of z₀ (length n). */
  m0: VectorLike | number
  /** Covariance of z₀ (n×n). */
  P0: MatrixLike | number
}

/** The model with every part as rows of numbers: the working form the filter, the smoother and EM share. */
export type Model = { A: Mat; C: Mat; Q: Mat; R: Mat; m0: Vec; P0: Mat }

/** Read a `StateSpaceModel` into its working form, checking that the shapes agree (throws naming `where`). */
export function parseModel(model: StateSpaceModel, where: string): Model {
  const A = toMat(model.A, where)
  const C = toMat(model.C, where)
  const Q = toMat(model.Q, where)
  const R = toMat(model.R, where)
  const m0 = typeof model.m0 === 'number' ? [model.m0] : toVec(model.m0, where)
  const P0 = toMat(model.P0, where)
  const n = A.length
  const m = C.length
  const ok =
    A.every((r) => r.length === n) &&
    C.every((r) => r.length === n) &&
    Q.length === n &&
    R.length === m &&
    R.every((r) => r.length === m) &&
    m0.length === n &&
    P0.length === n
  if (!ok) throw new Error(`${where}: inconsistent model shapes (A ${n}×${A[0]?.length}, C ${m}×${C[0]?.length})`)
  return { A, C, Q, R, m0, P0 }
}

/** The model as tensors, for returning a fitted model. */
export function modelTensors(m: Model): { A: Matrix; C: Matrix; Q: Matrix; R: Matrix; m0: Tensor; P0: Matrix } {
  return { A: matT(m.A), C: matT(m.C), Q: matT(m.Q), R: matT(m.R), m0: vecT(m.m0), P0: matT(m.P0) }
}
/**
 * Draw a trajectory of length T and its observations. Noise is drawn as S ε with S S ᵀ = Q (or R) from a symmetric
 * eigendecomposition, so covariances with deterministic components (zero rows) give exact zeros rather than NaN, as the
 * site's floored Cholesky gave. Returns z₁ … z_T ([T, n]) and y₁ … y_T ([T, m]) and the drawn z₀.
 */
export function simulateStateSpace(
  s: Stream,
  model: StateSpaceModel,
  T: number,
): { states: Matrix; observations: Matrix; initial: Tensor } {
  const { A, C, Q, R, m0, P0 } = parseModel(model, 'simulateStateSpace')
  const n = A.length
  const m = C.length
  const draw = (S: Mat, key: string, t: number) => matvec(S, toFlat(normals(child(s, key, t), S[0].length)))
  const Sq = sqrtPsd(Q).S
  const Sr = sqrtPsd(R).S
  let z = addV(m0, draw(sqrtPsd(P0).S, 'initial', 0))
  const initial = z
  const states: Vec[] = []
  const obs: Vec[] = []
  for (let t = 0; t < T; t++) {
    z = addV(matvec(A, z), draw(Sq, 'state', t))
    states.push(z)
    obs.push(addV(matvec(C, z), draw(Sr, 'observation', t)))
  }
  return { states: stackVecs(states, n), observations: stackVecs(obs, m), initial: vecT(initial) }
}

/** The Kalman filter's output, over t = 1 … T. */
export type KalmanFilterResult = {
  /** μ_{t|t−1} ([T, n]). */
  predictedMean: Matrix
  /** P_{t|t−1} ([T, n, n]). */
  predictedCov: Tensor
  /** μ_{t|t} ([T, n]). */
  mean: Matrix
  /** P_{t|t} ([T, n, n]). */
  cov: Tensor
  /** K_t ([T, n, m]); zero for a missing or singular step. */
  gain: Tensor
  /** Innovations y_t − C μ_{t|t−1} ([T, m]); NaN for a missing step. */
  innovation: Matrix
  /** Innovation covariances S_t = C P_{t|t−1} Cᵀ + R ([T, m, m]). */
  innovationCov: Tensor
  /** log p(y₁, …, y_T) = Σ log N(y_t; C μ_{t|t−1}, S_t) over observed steps. */
  logLikelihood: number
  /** Each step's term of the log-likelihood (0 for a missing step). */
  logLikelihoodTerms: Tensor
  /** Steps whose S_t was singular (the update was skipped and the step's likelihood term is NaN). */
  singularSteps: number[]
}

/** The filter's output as plain arrays over t = 1 … T, shared by the smoother and EM (`filterArrays`). */
export type FilterArrays = {
  mPred: Vec[]
  PPred: Mat[]
  m: Vec[]
  P: Mat[]
  K: Mat[]
  v: Vec[]
  S: Mat[]
  terms: number[]
  logLikelihood: number
  singularSteps: number[]
}

/**
 * The Kalman filter on the working form (see `kalmanFilter`, which packs this into tensors): `ys` is T rows of m
 * observations, NaN rows missing.
 */
export function filterArrays(md: Model, ys: Mat): FilterArrays {
  const { A, C, Q, R } = md
  const n = A.length
  const mDim = C.length
  const I = eyeM(n)
  let m = md.m0
  let P = md.P0
  const out: FilterArrays = {
    mPred: [],
    PPred: [],
    m: [],
    P: [],
    K: [],
    v: [],
    S: [],
    terms: [],
    logLikelihood: 0,
    singularSteps: [],
  }
  ys.forEach((y, t) => {
    const mPred = matvec(A, m)
    const PPred = symmetrise(addM(sandwich(A, P), Q))
    const S = symmetrise(addM(sandwich(C, PPred), R))
    const missing = y.some((v) => Number.isNaN(v))
    let K = zerosM(n, mDim)
    let v = new Array<number>(mDim).fill(NaN)
    let term = 0
    m = mPred
    P = PPred
    if (!missing) {
      v = subV(y, matvec(C, mPred))
      // K = P⁻ Cᵀ S⁻¹, from Kᵀ = S⁻¹ C P⁻ (S symmetric).
      const sol = solveM(S, matmul(C, PPred))
      if (sol.x) {
        K = transpose(sol.x)
        const Sv = solveM(
          S,
          v.map((x) => [x]),
        ).x!
        const quad = v.reduce((acc, vi, i) => acc + vi * Sv[i][0], 0)
        term = -0.5 * (mDim * Math.log(2 * Math.PI) + sol.logDet + quad)
        m = addV(mPred, matvec(K, v))
        // Joseph form (I − KC) P⁻ (I − KC)ᵀ + K R Kᵀ keeps P symmetric positive semi-definite in finite precision.
        const J = subM(I, matmul(K, C))
        P = symmetrise(addM(sandwich(J, PPred), sandwich(K, R)))
      } else {
        term = NaN
        out.singularSteps.push(t)
      }
    }
    out.mPred.push(mPred)
    out.PPred.push(PPred)
    out.m.push(m)
    out.P.push(P)
    out.K.push(K)
    out.v.push(v)
    out.S.push(S)
    out.terms.push(term)
    out.logLikelihood += term
  })
  return out
}

/** The filter's arrays as the public `KalmanFilterResult` tensors. */
export function packFilter(f: FilterArrays, n: number, m: number): KalmanFilterResult {
  return {
    predictedMean: stackVecs(f.mPred, n),
    predictedCov: stackMats(f.PPred, n, n),
    mean: stackVecs(f.m, n),
    cov: stackMats(f.P, n, n),
    gain: stackMats(f.K, n, m),
    innovation: stackVecs(f.v, m),
    innovationCov: stackMats(f.S, m, m),
    logLikelihood: f.logLikelihood,
    logLikelihoodTerms: vecT(f.terms),
    singularSteps: f.singularSteps,
  }
}

/**
 * The Kalman filter (Kalman, 1960). Predict μ⁻ = A μ, P⁻ = A P Aᵀ + Q; update with S = C P⁻ Cᵀ + R,
 * K = P⁻ Cᵀ S⁻¹, μ = μ⁻ + K(y − C μ⁻) and the covariance in Joseph form. `y` is [T, m] (or a length-T vector when
 * m = 1). No Cholesky factor is taken, so Q or R with zero rows filter normally; a singular S is reported in
 * `singularSteps`, never turned into NaN states.
 */
export function kalmanFilter(model: StateSpaceModel, y: VectorLike | MatrixLike): KalmanFilterResult {
  const md = parseModel(model, 'kalmanFilter')
  const ys = toSeries(y, 'kalmanFilter')
  if (ys.length && ys[0].length !== md.C.length)
    throw new Error(`kalmanFilter: observations have ${ys[0].length} columns, C has ${md.C.length} rows`)
  return packFilter(filterArrays(md, ys), md.A.length, md.C.length)
}

/** The RTS smoother's output: smoothed moments for t = 1 … T, and for z₀. */
export type SmootherResult = {
  /** μ_{t|T} ([T, n]). */
  mean: Matrix
  /** P_{t|T} ([T, n, n]). */
  cov: Tensor
  /** Smoother gains G_t = P_{t|t} Aᵀ P_{t+1|t}⁻¹ ([T, n, n]; the last is zero). */
  gain: Tensor
  /** Cov(z_t, z_{t−1} | y) = P_{t|T} G_{t−1}ᵀ ([T, n, n]; entry t pairs z_t with z_{t−1}, z₀ for the first). */
  lagOneCov: Tensor
  /** μ_{0|T} and P_{0|T}. */
  initialMean: Tensor
  initialCov: Matrix
  /** Steps whose predicted covariance P_{t+1|t} was singular; a pseudo-solve (ridge 1e-12·scale) was used there. */
  singularSteps: number[]
}

/** The RTS smoother's output as plain arrays (see `smootherArrays`). */
export type SmootherArrays = { m: Vec[]; P: Mat[]; G: Mat[]; lag: Mat[]; m0: Vec; P0: Mat; singularSteps: number[] }

/** G = P Aᵀ Ppred⁻¹, via Gᵀ = Ppred⁻¹ A P (Ppred symmetric), with a reported ridge if Ppred is singular. */
function smootherGain(P: Mat, A: Mat, PPred: Mat, t: number, singular: number[]): Mat {
  const rhs = matmul(A, P)
  let sol = solveM(PPred, rhs).x
  if (!sol) {
    singular.push(t)
    let scale = 0
    PPred.forEach((r, i) => (scale = Math.max(scale, Math.abs(r[i]))))
    const ridge = 1e-12 * (scale || 1)
    sol = solveM(addM(PPred, scaleM(eyeM(PPred.length), ridge)), rhs).x ?? zerosM(PPred.length, P.length)
  }
  return transpose(sol)
}

/** The RTS smoother on the working form, from the filter's arrays (see `rtsSmoother`). */
export function smootherArrays(md: Model, f: FilterArrays): SmootherArrays {
  const T = f.m.length
  const n = md.A.length
  const m: Vec[] = new Array(T)
  const P: Mat[] = new Array(T)
  const G: Mat[] = new Array(T).fill(zerosM(n))
  const lag: Mat[] = new Array(T)
  const singularSteps: number[] = []
  m[T - 1] = f.m[T - 1]
  P[T - 1] = f.P[T - 1]
  for (let t = T - 2; t >= 0; t--) {
    const Gt = smootherGain(f.P[t], md.A, f.PPred[t + 1], t, singularSteps)
    G[t] = Gt
    m[t] = addV(f.m[t], matvec(Gt, subV(m[t + 1], f.mPred[t + 1])))
    P[t] = symmetrise(addM(f.P[t], sandwich(Gt, subM(P[t + 1], f.PPred[t + 1]))))
  }
  // One more step back to z₀, whose "filtered" moments are the prior (m₀, P₀).
  const G0 = T > 0 ? smootherGain(md.P0, md.A, f.PPred[0], -1, singularSteps) : zerosM(n)
  const m0 = T > 0 ? addV(md.m0, matvec(G0, subV(m[0], f.mPred[0]))) : md.m0
  const P0 = T > 0 ? symmetrise(addM(md.P0, sandwich(G0, subM(P[0], f.PPred[0])))) : md.P0
  // Cov(z_t, z_{t−1} | y) = P_{t|T} G_{t−1}ᵀ (de Jong, 1989; Shumway & Stoffer, 2017, Property 6.3 in this form).
  for (let t = 0; t < T; t++) lag[t] = matmul(P[t], transpose(t === 0 ? G0 : G[t - 1]))
  return { m, P, G, lag, m0, P0, singularSteps }
}

/**
 * The Rauch–Tung–Striebel smoother (Rauch, Tung & Striebel, 1965): backwards from the last filtered estimate,
 * G_t = P_{t|t} Aᵀ P_{t+1|t}⁻¹, μ_{t|T} = μ_{t|t} + G_t(μ_{t+1|T} − μ_{t+1|t}),
 * P_{t|T} = P_{t|t} + G_t(P_{t+1|T} − P_{t+1|t})G_tᵀ. Also returns the lag-one covariances EM needs and the smoothed z₀.
 * Pass the filter's result or let it run the filter.
 */
export function rtsSmoother(model: StateSpaceModel, y: VectorLike | MatrixLike): SmootherResult {
  const md = parseModel(model, 'rtsSmoother')
  const ys = toSeries(y, 'rtsSmoother')
  const s = smootherArrays(md, filterArrays(md, ys))
  const n = md.A.length
  return {
    mean: stackVecs(s.m, n),
    cov: stackMats(s.P, n, n),
    gain: stackMats(s.G, n, n),
    lagOneCov: stackMats(s.lag, n, n),
    initialMean: vecT(s.m0),
    initialCov: matT(s.P0),
    singularSteps: s.singularSteps,
  }
}

/**
 * The steady-state Kalman filter: iterate the Riccati recursion P⁻ ← A (P⁻ − P⁻Cᵀ S⁻¹ C P⁻) Aᵀ + Q from P₀ until the
 * largest change is below `tolerance` relative to P, and return the limiting gain and covariances. `converged` is false
 * when `maxSteps` (default 10 000) ran out, e.g. for an undetectable unstable mode (Anderson & Moore, 1979,
 * "Optimal Filtering", §4.4).
 */
export function steadyStateKalman(
  model: StateSpaceModel,
  { tolerance = 1e-12, maxSteps = 10000 }: { tolerance?: number; maxSteps?: number } = {},
): { gain: Matrix; predictedCov: Matrix; cov: Matrix; iterations: number; converged: boolean } {
  const md = parseModel(model, 'steadyStateKalman')
  const { A, C, Q, R } = md
  const I = eyeM(A.length)
  let Pp = symmetrise(addM(sandwich(A, md.P0), Q))
  let K = zerosM(A.length, C.length)
  let Pf = Pp
  for (let it = 1; it <= maxSteps; it++) {
    const S = addM(sandwich(C, Pp), R)
    const sol = solveM(S, matmul(C, Pp)).x
    if (!sol) return { gain: matT(K), predictedCov: matT(Pp), cov: matT(Pf), iterations: it, converged: false }
    K = transpose(sol)
    const J = subM(I, matmul(K, C))
    Pf = symmetrise(addM(sandwich(J, Pp), sandwich(K, R)))
    const next = symmetrise(addM(sandwich(A, Pf), Q))
    let change = 0
    let size = 0
    next.forEach((r, i) =>
      r.forEach((v, j) => ((change = Math.max(change, Math.abs(v - Pp[i][j]))), (size = Math.max(size, Math.abs(v))))),
    )
    Pp = next
    if (change <= tolerance * (1 + size))
      return { gain: matT(K), predictedCov: matT(Pp), cov: matT(Pf), iterations: it, converged: true }
  }
  return { gain: matT(K), predictedCov: matT(Pp), cov: matT(Pf), iterations: maxSteps, converged: false }
}

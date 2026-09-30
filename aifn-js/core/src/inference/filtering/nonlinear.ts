/**
 * Nonlinear state-space filters for z_t = f(z_{t−1}) + w_t, y_t = h(z_t) + v_t with Gaussian noise: the extended
 * Kalman filter (linearise f and h by their Jacobians, from `aifn/foundation/autodiff`) and the unscented Kalman filter
 * (propagate sigma points through f and h). Both return the linear filter's result shape.
 */

import { jacobian } from 'aifn/foundation/autodiff'
import { isTensor, tensor, toFlat, type Tensor, type Value, type Vector } from 'aifn/foundation/tensor'
import { packFilter, type FilterArrays, type KalmanFilterResult } from './kalman'
import {
  addM,
  addV,
  matmul,
  matvec,
  outer,
  sandwich,
  scaleM,
  solveM,
  sqrtPsd,
  subM,
  subV,
  symmetrise,
  toMat,
  toSeries,
  toVec,
  transpose,
  zerosM,
  type Mat,
  type MatrixLike,
  type Vec,
  type VectorLike,
} from './mat'

/**
 * A nonlinear state-space model. `f` and `h` take the state as a length-n vector and return a vector (n and m long).
 * For the extended filter they must be written with `aifn/foundation/tensor` primitives so that `aifn/foundation/autodiff` can
 * differentiate them (e.g. `stack([add(get(z, 0), get(z, 1)), sin(get(z, 0))])`).
 */
export type NonlinearStateSpaceModel = {
  f: (z: Vector) => Value
  h: (z: Vector) => Value
  Q: MatrixLike | number
  R: MatrixLike | number
  m0: VectorLike | number
  P0: MatrixLike | number
}

const asVec = (v: Value, where: string): Vec => {
  if (typeof v === 'number') return [v]
  if (!isTensor(v)) throw new Error(`${where}: f and h must return numbers or tensors`)
  return toFlat(v)
}

function parse(model: NonlinearStateSpaceModel, where: string) {
  return {
    Q: toMat(model.Q, where),
    R: toMat(model.R, where),
    m0: typeof model.m0 === 'number' ? [model.m0] : toVec(model.m0, where),
    P0: toMat(model.P0, where),
  }
}

/** The update shared by both filters, given the predicted mean and covariance, the predicted observation ŷ, S and the cross-covariance. */
function update(
  out: FilterArrays,
  t: number,
  y: Vec,
  mPred: Vec,
  PPred: Mat,
  yHat: Vec,
  S: Mat,
  cross: Mat,
): { m: Vec; P: Mat } {
  const n = mPred.length
  const mDim = S.length
  const missing = y.some((v) => Number.isNaN(v))
  let K = zerosM(n, mDim)
  let v = new Array<number>(mDim).fill(NaN)
  let m = mPred
  let P = PPred
  let term = 0
  if (!missing) {
    v = subV(y, yHat)
    // K = Σ_zy S⁻¹ via Kᵀ = S⁻¹ Σ_zyᵀ.
    const sol = solveM(S, transpose(cross))
    if (sol.x) {
      K = transpose(sol.x)
      const Sv = solveM(
        S,
        v.map((x) => [x]),
      ).x!
      term = -0.5 * (mDim * Math.log(2 * Math.PI) + sol.logDet + v.reduce((a, vi, i) => a + vi * Sv[i][0], 0))
      m = addV(mPred, matvec(K, v))
      P = symmetrise(subM(PPred, sandwich(K, S)))
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
  return { m, P }
}

const empty = (): FilterArrays => ({
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
})

/**
 * The extended Kalman filter (Jazwinski, 1970; Särkkä, 2013, Algorithm 5.4): predict μ⁻ = f(μ), P⁻ = F P Fᵀ + Q with
 * F = ∂f/∂z at μ; update with H = ∂h/∂z at μ⁻, S = H P⁻ Hᵀ + R, K = P⁻ Hᵀ S⁻¹. Jacobians come from
 * `aifn/foundation/autodiff`'s `jacobian`. The log-likelihood is that of the linearised model.
 */
export function extendedKalmanFilter(model: NonlinearStateSpaceModel, y: VectorLike | MatrixLike): KalmanFilterResult {
  const { Q, R, m0, P0 } = parse(model, 'extendedKalmanFilter')
  const ys = toSeries(y, 'extendedKalmanFilter')
  const jf = jacobian(model.f as (z: Value) => Value)
  const jh = jacobian(model.h as (z: Value) => Value)
  // The Jacobian has shape [...shape of f(z), n]; read it as rows × n whatever the output's rank.
  const jac = (J: Value, rows: number, cols: number): Mat => {
    const flat = typeof J === 'number' ? [J] : toFlat(J as Tensor)
    return Array.from({ length: rows }, (_, i) => flat.slice(i * cols, (i + 1) * cols))
  }
  const out = empty()
  let m = m0
  let P = P0
  const n = m0.length
  ys.forEach((yt, t) => {
    const zt = tensor(m)
    const mPred = asVec(model.f(zt), 'extendedKalmanFilter')
    const F = jac(jf(zt), n, n)
    const PPred = symmetrise(addM(sandwich(F, P), Q))
    const zp = tensor(mPred)
    const yHat = asVec(model.h(zp), 'extendedKalmanFilter')
    const H = jac(jh(zp), yHat.length, n)
    const S = symmetrise(addM(sandwich(H, PPred), R))
    ;({ m, P } = update(out, t, yt, mPred, PPred, yHat, S, matmul(PPred, transpose(H))))
  })
  return packFilter(out, n, R.length)
}

/** Options of the unscented transform (Wan & van der Merwe, 2000): spread α, prior-knowledge β and secondary κ. */
export type UnscentedOptions = { alpha?: number; beta?: number; kappa?: number }

/**
 * The unscented Kalman filter (Julier & Uhlmann, 1997; Wan & van der Merwe, 2000): 2n + 1 sigma points
 * μ, μ ± √(n + λ) S_i with S Sᵀ = P and λ = α²(n + κ) − n, pushed through f (predict) and h (update), with mean
 * weights W₀ = λ/(n + λ), covariance weight W₀ + 1 − α² + β and 1/(2(n + λ)) for the rest. Defaults α = 1, β = 2,
 * κ = 0 (positive weights). The square root is a symmetric eigen-root, which exists for singular P.
 */
export function unscentedKalmanFilter(
  model: NonlinearStateSpaceModel,
  y: VectorLike | MatrixLike,
  { alpha = 1, beta = 2, kappa = 0 }: UnscentedOptions = {},
): KalmanFilterResult {
  const { Q, R, m0, P0 } = parse(model, 'unscentedKalmanFilter')
  const ys = toSeries(y, 'unscentedKalmanFilter')
  const n = m0.length
  const lambda = alpha * alpha * (n + kappa) - n
  const wm = [lambda / (n + lambda), ...new Array(2 * n).fill(1 / (2 * (n + lambda)))]
  const wc = [wm[0] + 1 - alpha * alpha + beta, ...wm.slice(1)]
  const sigma = (m: Vec, P: Mat): Vec[] => {
    const S = scaleM(sqrtPsd(P).S, Math.sqrt(n + lambda))
    const cols = transpose(S)
    return [m, ...cols.map((c) => addV(m, c)), ...cols.map((c) => subV(m, c))]
  }
  const moments = (pts: Vec[]) => {
    const mean = pts[0].map((_, i) => pts.reduce((s, p, k) => s + wm[k] * p[i], 0))
    return mean
  }
  const cov = (a: Vec[], ma: Vec, b: Vec[], mb: Vec): Mat =>
    a.reduce((acc, p, k) => addM(acc, scaleM(outer(subV(p, ma), subV(b[k], mb)), wc[k])), zerosM(ma.length, mb.length))
  const out = empty()
  let m = m0
  let P = P0
  const through = (g: (z: Vector) => Value, pts: Vec[]) => pts.map((p) => asVec(g(tensor(p)), 'unscentedKalmanFilter'))
  ys.forEach((yt, t) => {
    const fx = through(model.f, sigma(m, P))
    const mPred = moments(fx)
    const PPred = symmetrise(addM(cov(fx, mPred, fx, mPred), Q))
    const pts = sigma(mPred, PPred)
    const hx = through(model.h, pts)
    const yHat = moments(hx)
    const S = symmetrise(addM(cov(hx, yHat, hx, yHat), R))
    ;({ m, P } = update(out, t, yt, mPred, PPred, yHat, S, cov(pts, mPred, hx, yHat)))
  })
  return packFilter(out, n, R.length)
}

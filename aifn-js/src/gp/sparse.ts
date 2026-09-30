/**
 * Sparse Gaussian-process regression through m inducing inputs Z: the Nyström approximation Q = K_nm K_mm⁻¹ K_mn of
 * the training covariance, used three ways (Quiñonero-Candela and Rasmussen, 2005, "A unifying view of sparse
 * approximate Gaussian process regression"):
 *
 * - `sor` (subset of regressors) and `dtc` (deterministic training conditional) share the marginal likelihood
 *   N(y | m, Q + σ²I); they differ only in the predictive variance (SoR's collapses away from Z, DTC's does not).
 * - `fitc` (fully independent training conditional; Snelson and Ghahramani, 2006) corrects the diagonal:
 *   N(y | m, Q + diag(K − Q) + σ²I).
 * - `vfe` (Titsias, 2009, "Variational learning of inducing variables in sparse Gaussian processes") keeps the DTC
 *   likelihood term and subtracts tr(K − Q)/(2σ²), so the bound never exceeds the exact log marginal likelihood and
 *   inducing inputs can be optimised without overfitting.
 *
 * The algebra is the numerically stable form of GPflow's `SGPR` and `GPRFITC` (Matthews et al., 2017): with
 * L = chol(K_mm), A = L⁻¹K_mn Λ^{−½} and L_B = chol(I + AAᵀ), everything costs O(nm²). Every quantity is built from
 * tensor primitives, so the bound is differentiable in Z, the kernel's hyperparameters and σ².
 */

import { valueAndGrad } from 'aifn/autodiff'
import {
  asRows,
  gram,
  kernelDiagonal,
  kernelFromLog,
  parameterVector,
  type Kernel,
  type KernelParams,
} from 'aifn/kernels'
import { cholesky, solveTriangular } from 'aifn/linalg'
import { lbfgs, type LbfgsState } from 'aifn/optim'
import {
  add,
  diagonal,
  div,
  exp,
  fromData,
  get,
  log,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  slice,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn/tensor'
import { trace, type Trace } from 'aifn/trace'

const LOG_2PI = Math.log(2 * Math.PI)

/** The sparse approximation: `vfe` (Titsias), `fitc`, `dtc` or `sor`. */
export type SparseMethod = 'vfe' | 'fitc' | 'dtc' | 'sor'

/** Options of `sparseGp`. */
export type SparseGpOptions = {
  method?: SparseMethod
  /** Observation-noise variance σ² > 0. */
  noiseVariance: Value
  mean?: number
  /** Diagonal jitter on K_mm relative to its mean diagonal (default 1e-8), always added and reported. */
  relativeJitter?: number
}

/** The pieces of a sparse GP, possibly traced. */
type Pieces = {
  L: Value
  LB: Value
  c: Value
  bound: Value
  /** The bound's terms. */
  terms: { fit: Value; complexity: Value; trace: Value; constant: number }
  jitter: number
}

function eye(n: number): Tensor {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) out[i * n + i] = 1
  return fromData(out, [n, n])
}

function pieces(kernel: Kernel, x: Value, y: Tensor, z: Value, o: SparseGpOptions): Pieces {
  const { method = 'vfe', noiseVariance, mean = 0, relativeJitter = 1e-8 } = o
  const X = asRows(x)
  const Z = asRows(z)
  const n = shapeOfValue(X)[0]
  const m = shapeOfValue(Z)[0]
  const Kmm = gram(kernel, Z)
  const diagMean = (toFlat(unwrap(kernelDiagonal(kernel, Z)) as Tensor) as number[]).reduce((a, b) => a + b, 0) / m
  const jitter = relativeJitter * (diagMean > 0 ? diagMean : 1)
  const factor = cholesky(add(Kmm, mul(jitter, eye(m))))
  const L = factor.L
  const A0 = solveTriangular(L, gram(kernel, Z, X)) // L⁻¹K_mn, [m, n]
  const qnn = sum(square(A0), 0) // diag Q, [n]
  const knn = kernelDiagonal(kernel, X)
  const lambda =
    method === 'fitc'
      ? add(sub(knn, qnn), noiseVariance)
      : mul(noiseVariance, fromData(new Float64Array(n).fill(1), [n]))
  const rootLambda = sqrt(lambda)
  const A = div(A0, reshape(rootLambda, [1, n]))
  const { L: LB } = cholesky(add(eye(m), matmul(A, transpose(A))), { jitter: false })
  const yw = div(sub(y, mean), rootLambda)
  const c = solveTriangular(LB, matmul(A, yw))
  const constant = -0.5 * n * LOG_2PI
  const complexity = add(mul(-1, sum(log(diagonal(LB)))), mul(-0.5, sum(log(lambda))))
  const fit = add(mul(-0.5, sum(square(yw))), mul(0.5, sum(square(c))))
  const traceTerm = method === 'vfe' ? mul(-0.5, div(sum(sub(knn, qnn)), noiseVariance)) : 0
  return {
    L,
    LB,
    c,
    bound: add(add(add(fit, complexity), traceTerm), constant),
    terms: { fit, complexity, trace: traceTerm, constant },
    jitter: jitter + factor.jitter,
  }
}

/** A fitted sparse GP. */
export interface SparseGp<P extends KernelParams = KernelParams> {
  readonly method: SparseMethod
  readonly kernel: Kernel<P>
  /** Inducing inputs Z [m, d]. */
  readonly inducing: Tensor
  readonly noiseVariance: number
  readonly mean: number
  /**
   * The approximate log marginal likelihood (for `vfe`, the evidence lower bound), and its terms: data fit, complexity
   * (−log|L_B| − ½ Σ log Λ), the VFE trace penalty −tr(K − Q)/(2σ²) (0 for the others) and the constant.
   */
  readonly logMarginal: number
  readonly terms: { fit: number; complexity: number; trace: number; constant: number }
  /** Jitter added to K_mm. */
  readonly jitter: number
  /** Predictive mean and variance of f at xs [s, d] (or [s]); `noise` adds σ². */
  predict(xs: Tensor, options?: { noise?: boolean }): { mean: Tensor; variance: Tensor }
}

/**
 * A sparse GP regression with inducing inputs z [m, d] (or [m]) for inputs x [n, d] and targets y [n]. See the module
 * comment for the methods. With z = x, `fitc` and `vfe` reproduce the exact GP.
 */
export function sparseGp<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  z: Tensor,
  options: SparseGpOptions & { noiseVariance: number },
): SparseGp<P> {
  const { method = 'vfe', noiseVariance, mean = 0 } = options
  if (!(noiseVariance > 0)) throw new Error('sparseGp: noiseVariance must be positive')
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const Z = asRows(z) as Tensor
  const p = pieces(kernel, x, Y, Z, { ...options, method })
  const L = unwrap(p.L) as Tensor
  const LB = unwrap(p.LB) as Tensor
  const c = unwrap(p.c) as Tensor
  const num = (v: Value) => unwrap(v) as number
  return {
    method,
    kernel,
    inducing: Z,
    noiseVariance,
    mean,
    logMarginal: num(p.bound),
    terms: {
      fit: num(p.terms.fit),
      complexity: num(p.terms.complexity),
      trace: num(p.terms.trace),
      constant: p.terms.constant,
    },
    jitter: p.jitter,
    predict: (xs, { noise = false } = {}) => {
      const S = asRows(xs) as Tensor
      const t1 = solveTriangular(L, gram(kernel, Z, S)) as Tensor // [m, s]
      const t2 = solveTriangular(LB, t1) as Tensor
      const mu = add(matmul(c, t2), mean) as Tensor
      const q = toFlat(sum(square(t1), 0) as Tensor)
      const r = toFlat(sum(square(t2), 0) as Tensor)
      const k = toFlat(kernelDiagonal(kernel, S) as Tensor)
      const extra = noise ? noiseVariance : 0
      // SoR's predictive covariance is Q_** − Q_*m(…)Q_m* = t2ᵀt2; the others keep K_** − Q_** as well.
      const variance = Float64Array.from(k, (kk, j) => (method === 'sor' ? r[j] : kk - q[j] + r[j]) + extra)
      return { mean: mu, variance: fromData(variance, [variance.length]) }
    },
  }
}

/** The approximate log marginal likelihood (a traced value when its arguments are traced). */
export function sparseLogMarginal(kernel: Kernel, x: Value, y: Tensor, z: Value, options: SparseGpOptions): Value {
  return pieces(kernel, x, y, z, options).bound
}

/** Options of `fitSparseGp`. */
export type FitSparseGpOptions = {
  method?: SparseMethod
  /** Starting noise variance (default 0.1). */
  noiseVariance?: number
  fitNoise?: boolean
  /** Move the inducing inputs too (default true). */
  fitInducing?: boolean
  /** Fit the kernel's hyperparameters (default true); otherwise they stay as given. */
  fitKernel?: boolean
  mean?: number
  maxSteps?: number
  tolerance?: number
}

/** The result of `fitSparseGp`. */
export type SparseGpFit<P extends KernelParams = KernelParams> = {
  model: SparseGp<P>
  /** The L-BFGS trace over [log θ, log σ², Z (flattened)], each part present when it is fitted. */
  training: Trace<LbfgsState>
  converged: boolean
}

/**
 * Maximise the sparse approximation's log marginal likelihood (the ELBO for `vfe`) over the kernel's log
 * hyperparameters, log σ² and the inducing inputs, by L-BFGS with gradients from `aifn/autodiff`.
 */
export function fitSparseGp<P extends KernelParams>(
  kernel: Kernel<P>,
  x: Tensor,
  y: Tensor,
  z: Tensor,
  options: FitSparseGpOptions = {},
): SparseGpFit<P> {
  const {
    method = 'vfe',
    noiseVariance = 0.1,
    fitNoise = true,
    fitInducing = true,
    fitKernel = true,
    mean = 0,
  } = options
  const { maxSteps = 200, tolerance = 1e-6 } = options
  const X = asRows(x) as Tensor
  const Y = y.shape.length === 2 ? (reshape(y, [y.shape[0]]) as Tensor) : y
  const Z0 = asRows(z) as Tensor
  const [m, d] = Z0.shape
  const pv = parameterVector(kernel)
  const k = fitKernel ? pv.values.length : 0
  const zStart = k + (fitNoise ? 1 : 0)
  const unpack = (theta: Value) => ({
    kernel: fitKernel ? kernelFromLog(kernel, slice(theta, [0, k])) : kernel,
    noise: fitNoise ? exp(get(theta, k)) : noiseVariance,
    z: fitInducing ? reshape(slice(theta, [zStart, zStart + m * d]), [m, d]) : Z0,
  })
  const negative = valueAndGrad((theta: Tensor) => {
    const u = unpack(theta)
    return mul(-1, sparseLogMarginal(u.kernel, X, Y, u.z, { method, noiseVariance: u.noise, mean }))
  })
  const objective = (theta: Tensor) => {
    try {
      const { value, grad } = negative(theta)
      if (!Number.isFinite(value as number)) return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
      return { value: value as number, grad: grad as Tensor }
    } catch {
      return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
    }
  }
  const start = [
    ...(fitKernel ? Array.from(pv.values, Math.log) : []),
    ...(fitNoise ? [Math.log(noiseVariance)] : []),
    ...(fitInducing ? (toFlat(Z0) as number[]) : []),
  ]
  const training = trace(lbfgs(objective, { tolerance }), { x0: tensor(start) }, maxSteps, {
    record: { logMarginal: (s: LbfgsState) => -s.value },
  })
  const final = training.steps[training.steps.length - 1]
  const u = unpack(final.x)
  const model = sparseGp(u.kernel as Kernel<P>, X, Y, unwrap(u.z) as Tensor, {
    method,
    noiseVariance: unwrap(u.noise) as number,
    mean,
  })
  return { model, training, converged: final.converged }
}

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

import { valueAndGrad } from 'aifn/foundation/autodiff'
import { asRows, gram, kernelDiagonal, kernelFromLog, type Kernel, type KernelParams } from 'aifn/learning/kernels'
import { ravel } from 'aifn/foundation/pytree'
import { permutation, type Stream } from 'aifn/foundation/random'
import {
  withExpectation,
  type Decides,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Supervised,
  type Trained,
} from 'aifn/learning/estimators'
import { Normal, type Univariate } from 'aifn/probability/distributions'
import { kernelLogVector } from './regression'
import { cholesky, solveTriangular } from 'aifn/numerics/linalg'
import { lbfgs, type LbfgsState } from 'aifn/optim/second-order'
import {
  add,
  diagonal,
  div,
  exp,
  fromData,
  log,
  matmul,
  mul,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sub,
  sum,
  take,
  tensor,
  toFlat,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn/foundation/tensor'
import { trace, type Trace } from 'aifn/foundation/trace'
import { defineModel } from 'aifn/learning/estimators'
import { bool, int, oneOf, real, space } from 'aifn/foundation/space'

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
 * hyperparameters, log σ² and the inducing inputs, by L-BFGS with gradients from `aifn/foundation/autodiff`.
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
  // θ = [log kernel hyperparameters (pytree order), log σ², Z flattened], each part present when it is fitted. The
  // gradient is taken with respect to the parts (the kernel's as a pytree) and concatenated in the same order.
  const lv = kernelLogVector(kernel)
  const k = fitKernel ? lv.vector.length : 0
  const zStart = k + (fitNoise ? 1 : 0)
  const parts = (theta: ArrayLike<number>) => ({
    tree: fitKernel ? lv.unravel(Array.from(theta).slice(0, k)) : lv.unravel(lv.vector),
    logNoise: fitNoise ? theta[k] : Math.log(noiseVariance),
    z: fitInducing ? fromData(Float64Array.from(Array.from(theta).slice(zStart, zStart + m * d)), [m, d]) : Z0,
  })
  const negative = valueAndGrad(
    (tree: P, logNoise: Value, z: Value) =>
      mul(-1, sparseLogMarginal(kernelFromLog(kernel, tree), X, Y, z, { method, noiseVariance: exp(logNoise), mean })),
    { argnums: [0, 1, 2] },
  )
  const objective = (theta: Tensor) => {
    try {
      const u = parts(toFlat(theta))
      const { value, grad } = negative(u.tree, u.logNoise, u.z)
      if (!Number.isFinite(value as number)) return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
      const [gTree, gNoise, gZ] = grad as [P, Value, Value]
      const g = [
        ...(fitKernel ? ravel(gTree).vector : []),
        ...(fitNoise ? [typeof gNoise === 'number' ? gNoise : toFlat(gNoise as Tensor)[0]] : []),
        ...(fitInducing ? toFlat(gZ as Tensor) : []),
      ]
      return { value: value as number, grad: fromData(Float64Array.from(g), [g.length]) }
    } catch {
      return { value: Infinity, grad: new Float64Array(theta.shape[0]) }
    }
  }
  const start = [
    ...(fitKernel ? lv.vector : []),
    ...(fitNoise ? [Math.log(noiseVariance)] : []),
    ...(fitInducing ? (toFlat(Z0) as number[]) : []),
  ]
  const training = trace(lbfgs(objective, { tolerance }), { x0: tensor(start) }, maxSteps, {
    record: { logMarginal: (s: LbfgsState) => -s.value },
  })
  const final = training.final
  const u = parts(toFlat(final.x))
  const model = sparseGp(kernelFromLog(kernel, u.tree), X, Y, u.z, {
    method,
    noiseVariance: Math.exp(u.logNoise),
    mean,
  })
  return { model, training, converged: final.converged }
}

// ── Estimator ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `sparseGaussianProcessRegressor`. */
export type SparseGaussianProcessRegressorParams<P extends KernelParams = KernelParams> = FitSparseGpOptions & {
  kernel: Kernel<P>
  /** Number of inducing inputs m, chosen from the training inputs without replacement (default min(n, 20)). */
  inducing?: number
  /** Fit the kernel, noise and inducing inputs by maximising the bound (default true). */
  optimise?: boolean
}

/** A fitted sparse GP regression model. */
export interface SparseGaussianProcessRegressionModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Partial<Trained<LbfgsState>> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'sparse-gaussian-process-regression'
  readonly sparse: SparseGp<P>
}

/**
 * Sparse GP regression as an estimator (the C1 conformance of `sparseGp`/`fitSparseGp`): the inducing inputs start
 * at `inducing` training inputs drawn from `options.stream` (the first ones without a stream), and with `optimise` the
 * kernel, noise and inducing inputs are fitted by `fitSparseGp`, whose L-BFGS run is kept in `training`.
 * Capabilities: `forward` and `decide` (the predictive mean), `predictive` (normals N(mean, var + σ²)), `expect`.
 */
export function sparseGaussianProcessRegressor<P extends KernelParams>(
  params: SparseGaussianProcessRegressorParams<P>,
): Estimator<Supervised<Tensor, Tensor>, SparseGaussianProcessRegressionModel<P>> {
  const { kernel, optimise = true, method = 'vfe', noiseVariance = 0.1, mean = 0 } = params
  return {
    name: 'sparse-gaussian-process-regression',
    params,
    fit({ x, y }, options: { stream?: Stream } = {}) {
      const X = asRows(x) as Tensor
      const n = X.shape[0]
      const m = Math.min(n, params.inducing ?? 20)
      const rows = options.stream
        ? Array.from(toFlat(permutation(options.stream, n))).slice(0, m)
        : [...Array(m).keys()]
      const Z = take(X, rows) as Tensor
      let sparse: SparseGp<P>
      let training: Trace<LbfgsState> | undefined
      if (optimise) {
        const fitted = fitSparseGp(kernel, X, y, Z, params)
        sparse = fitted.model
        training = fitted.training
      } else sparse = sparseGp(kernel, X, y, Z, { method, noiseVariance, mean })
      const forward = (input: Tensor) => sparse.predict(input).mean
      const base = {
        kind: 'model' as const,
        name: 'sparse-gaussian-process-regression' as const,
        sparse,
        ...(training ? { training } : {}),
        forward,
        decide: forward,
        predictive: (input: Tensor) => {
          const p = sparse.predict(input, { noise: true })
          return Normal(
            p.mean,
            fromData(Float64Array.from(toFlat(p.variance), Math.sqrt), p.variance.shape),
          ) as Univariate<Tensor>
        },
      }
      return withExpectation(base)
    },
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'sparseGaussianProcessRegressor',
    module: 'learning/gaussian-processes',
    name: 'Sparse Gaussian process regression',
    summary: 'Inducing-point GP regression (VFE, FITC or DTC); the kernel is a required argument.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect'],
    hyper: space({
      inducing: int(1, 500, { default: 20 }),
      method: oneOf(['vfe', 'fitc', 'dtc', 'sor']),
      noiseVariance: real(1e-6, 10, { default: 0.1, scale: 'log' }),
      optimise: bool({ default: true }),
      mean: real(-10, 10, { default: 0 }),
    }),
    notes: ['sparse-gaussian-processes'],
    cite: ['titsias2009'],
  },
  sparseGaussianProcessRegressor,
)

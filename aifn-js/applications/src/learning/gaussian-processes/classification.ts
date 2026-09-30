/**
 * Binary Gaussian-process classification by the Laplace approximation: Newton's method for the posterior mode of the
 * latent function (a traceable `Algorithm`), the approximate log marginal likelihood, and predictive probabilities.
 *
 * Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", Algorithms 3.1 (mode finding, the stable
 * form with B = I + W^½KW^½) and 3.2 (predictions), and eq. 3.32 (the approximate log marginal likelihood).
 */

import { Bernoulli, type Univariate } from 'aifn/probability/distributions'
import type { Status } from 'aifn/foundation/contracts'
import { gaussHermite } from 'aifn/numerics/quadrature'
import {
  withExpectation,
  withSampling,
  type Decides,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Scores,
  type Supervised,
  type Trained,
} from 'aifn/learning/estimators'
import { asRows, gram, kernelDiagonal, type Kernel, type KernelParams } from 'aifn/learning/kernels'
import { cholesky, solveTriangular } from 'aifn/numerics/linalg'
import { normalCdf, normalLogCdf, normalPdf, sigmoid, softplus } from 'aifn/numerics/special'
import { fromData, matmul, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'

/** The link from latent f to P(y = 1 | f): logistic σ(f) or probit Φ(f). */
export type ClassificationLikelihood = 'logistic' | 'probit'

/** log p(y | f), its gradient and the negative Hessian diagonal W, for labels t ∈ {0, 1} (y = 2t − 1). */
function likelihoodTerms(likelihood: ClassificationLikelihood, t: Float64Array, f: Float64Array) {
  const n = f.length
  const grad = new Float64Array(n)
  const W = new Float64Array(n)
  let logLik = 0
  for (let i = 0; i < n; i++) {
    const y = 2 * t[i] - 1
    if (likelihood === 'logistic') {
      // log σ(yf) = −softplus(−yf); ∇ = t − σ(f); W = σ(f)(1 − σ(f)) (R&W eqs. 3.15).
      logLik -= softplus(-y * f[i])
      const p = sigmoid(f[i])
      grad[i] = t[i] - p
      W[i] = p * (1 - p)
    } else {
      // log Φ(yf); ∇ = yN(f)/Φ(yf); W = r² + yf·r with r = N(f)/Φ(yf) (R&W eq. 3.16), r from logs for the tails.
      const z = y * f[i]
      const logPhi = normalLogCdf(z)
      logLik += logPhi
      const r = Math.exp(Math.log(normalPdf(z)) - logPhi)
      grad[i] = y * r
      W[i] = r * r + z * r
    }
  }
  return { logLik, grad, W }
}

/** A state of the Laplace mode search. */
export type LaplaceState = Status & {
  /** Newton steps taken. */
  t: number
  /** Latent values f at the training inputs, [n]. */
  f: Tensor
  /** a with f = Ka (R&W Algorithm 3.1). */
  a: Tensor
  /** Ψ(f) = log p(y | f) − ½ fᵀK⁻¹f, the objective Newton's method increases. */
  objective: number
  /** The approximate log marginal likelihood at f (eq. 3.32). */
  logMarginal: number
  converged: boolean
}

/** The problem a Laplace mode search solves. */
export type LaplaceProblem = {
  K: Tensor
  /** Labels 0 or 1, [n]. */
  labels: Tensor
  likelihood: ClassificationLikelihood
  /** Stop when Ψ rises by less than this (default 1e-10, as scikit-learn). */
  tolerance?: number
}

/** Quantities at f: W, B's factor and the approximate log marginal likelihood. */
function at(problem: LaplaceProblem, f: Float64Array) {
  const t = Float64Array.from(toFlat(problem.labels))
  const n = f.length
  const { logLik, grad, W } = likelihoodTerms(problem.likelihood, t, f)
  const sW = W.map(Math.sqrt)
  const K = Float64Array.from(toFlat(problem.K))
  const B = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) B[i * n + j] = (i === j ? 1 : 0) + sW[i] * K[i * n + j] * sW[j]
  const { L, jitter } = cholesky(fromData(B, [n, n]))
  const Lf = Float64Array.from(toFlat(L))
  let logDetHalf = 0
  for (let i = 0; i < n; i++) logDetHalf += Math.log(Lf[i * n + i])
  return { logLik, grad, W, sW, K, L, jitter, logDetHalf }
}

/**
 * Newton's method for the mode of p(f | X, y) under a GP prior with Gram matrix K and a logistic or probit likelihood
 * (Rasmussen and Williams, 2006, Algorithm 3.1). Each step is b = Wf + ∇log p(y | f),
 * a = b − W^½ L⁻ᵀ L⁻¹ W^½ K b, f = Ka, with L = chol(I + W^½ K W^½). The objective is concave, so plain Newton steps
 * converge; `init` takes an optional starting f (default 0).
 */
export function laplaceMode(problem: LaplaceProblem): Algorithm<{ f?: Tensor }, LaplaceState> {
  const tol = problem.tolerance ?? 1e-10
  const n = problem.K.shape[0]
  const stateAt = (f: Float64Array, a: Float64Array, t: number, previous: number): LaplaceState => {
    const q = at(problem, f)
    let fa = 0
    for (let i = 0; i < n; i++) fa += f[i] * a[i]
    const objective = q.logLik - 0.5 * fa
    return {
      t,
      f: fromData(f, [n]),
      a: fromData(a, [n]),
      objective,
      logMarginal: objective - q.logDetHalf,
      converged: t > 0 && Math.abs(objective - previous) < tol,
      diverged: !Number.isFinite(objective),
    }
  }
  return {
    name: 'gp-laplace-mode',
    init: ({ f } = {}) => {
      const f0 = f ? Float64Array.from(toFlat(f)) : new Float64Array(n)
      // a = K⁻¹f; from 0 it is 0, otherwise solve through the factor of K (+ jitter).
      let a0 = new Float64Array(n)
      if (f) {
        const { L } = cholesky(problem.K)
        a0 = Float64Array.from(toFlat(solveTriangular(L, solveTriangular(L, f), { transpose: true }) as Tensor))
      }
      return stateAt(f0, a0, 0, -Infinity)
    },
    step: (state) => {
      const f = Float64Array.from(toFlat(state.f))
      const q = at(problem, f)
      const b = Float64Array.from(f, (fi, i) => q.W[i] * fi + q.grad[i])
      // W^½ K b
      const Kb = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += q.K[i * n + j] * b[j]
        Kb[i] = q.sW[i] * s
      }
      const inner = toFlat(solveTriangular(q.L, solveTriangular(q.L, fromData(Kb, [n])), { transpose: true }) as Tensor)
      const a = Float64Array.from(b, (bi, i) => bi - q.sW[i] * inner[i])
      const fNew = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += q.K[i * n + j] * a[j]
        fNew[i] = s
      }
      return stateAt(fNew, a, state.t + 1, state.objective)
    },
  }
}

/** Options of `gpClassifier`. */
export type GpClassifierParams<P extends KernelParams = KernelParams> = {
  kernel: Kernel<P>
  likelihood?: ClassificationLikelihood
  /** Most Newton steps (default 100). */
  maxSteps?: number
  /** Newton convergence tolerance on Ψ (default 1e-10). */
  tolerance?: number
}

/** A fitted Laplace GP classifier. */
export interface GpClassifierModel<P extends KernelParams = KernelParams>
  extends
    Fitted<Tensor, Tensor>,
    Scores<Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Univariate<Tensor>>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<LaplaceState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gp-classifier'
  readonly kernel: Kernel<P>
  readonly likelihood: ClassificationLikelihood
  /** The posterior mode f̂ at the training inputs, [n]. */
  readonly mode: Tensor
  /** The Laplace approximation to log p(y | X). */
  readonly logMarginal: number
  readonly converged: boolean
  /** Mean and variance of the approximate latent posterior q(f* | y) at xs. */
  latent(xs: Tensor): { mean: Tensor; variance: Tensor }
}

/**
 * Binary GP classification (labels 0 and 1) by the Laplace approximation. Capabilities: `forward` and `score` (the
 * latent mean E_q[f*]), `predictive` (Bernoulli with π* = ∫ σ(f) q(f*) df*, exact Φ(μ/√(1 + v)) for the probit link
 * and 32-point Gauss–Hermite quadrature for the logistic one; R&W eq. 3.25 and Algorithm 3.2), `decide` (π* > ½),
 * `expect`, `sample`. The Newton run is kept in `training`.
 */
export function gpClassifier<P extends KernelParams>(
  params: GpClassifierParams<P>,
): Estimator<Supervised<Tensor, Tensor>, GpClassifierModel<P>> {
  const { kernel, likelihood = 'logistic', maxSteps = 100, tolerance = 1e-10 } = params
  // 32-point Gauss–Hermite (probabilists') rule, weights normalised to sum to 1: E[g(Z)], Z ~ N(0, 1).
  const rule = gaussHermite(32, { probabilists: true })
  const nodes = toFlat(rule.nodes)
  const ruleWeights = toFlat(rule.weights)
  const total = ruleWeights.reduce((a, b) => a + b, 0)
  const weights = ruleWeights.map((w) => w / total)
  return {
    name: 'gp-classifier',
    params,
    fit({ x, y }, options: FitOptions = {}) {
      const X = asRows(x) as Tensor
      const n = X.shape[0]
      const t = Float64Array.from(toFlat(y))
      if (t.length !== n) throw new Error(`gpClassifier: ${n} inputs but ${t.length} labels`)
      if (!t.every((v) => v === 0 || v === 1)) throw new Error('gpClassifier: labels must be 0 or 1')
      const K = gram(kernel, X) as Tensor
      const problem: LaplaceProblem = { K, labels: fromData(t, [n]), likelihood, tolerance }
      const training: Trace<LaplaceState> = trace(laplaceMode(problem), {}, maxSteps, {
        every: options.trace?.every ?? 1,
        record: { objective: (s) => s.objective, logMarginal: (s) => s.logMarginal },
      })
      const final = training.final
      const f = Float64Array.from(toFlat(final.f))
      const q = at(problem, f)
      const latent = (xs: Tensor) => {
        const S = asRows(xs) as Tensor
        const Ks = gram(kernel, X, S) as Tensor // [n, m]
        const mean = matmul(fromData(q.grad, [n]), Ks) as Tensor
        const m = S.shape[0]
        const ks = Float64Array.from(toFlat(Ks))
        const scaled = new Float64Array(n * m)
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) scaled[i * m + j] = q.sW[i] * ks[i * m + j]
        const V = toFlat(solveTriangular(q.L, fromData(scaled, [n, m])) as Tensor)
        const kss = toFlat(kernelDiagonal(kernel, S) as Tensor)
        const variance = Float64Array.from(kss, (k, j) => {
          let s = 0
          for (let i = 0; i < n; i++) s += V[i * m + j] ** 2
          return Math.max(k - s, 0)
        })
        return { mean, variance: fromData(variance, [m]) }
      }
      const probability = (xs: Tensor) => {
        const { mean, variance } = latent(xs)
        const mu = toFlat(mean)
        const v = toFlat(variance)
        return fromData(
          Float64Array.from(mu, (m, j) => {
            if (likelihood === 'probit') return normalCdf(m / Math.sqrt(1 + v[j]))
            let s = 0
            for (let k = 0; k < nodes.length; k++) s += weights[k] * sigmoid(m + Math.sqrt(v[j]) * nodes[k])
            return s
          }),
          [mu.length],
        )
      }
      const forward = (xs: Tensor) => latent(xs).mean
      const base = {
        kind: 'model' as const,
        name: 'gp-classifier' as const,
        kernel,
        likelihood,
        mode: final.f,
        logMarginal: final.logMarginal,
        converged: final.converged,
        training,
        latent,
        forward,
        score: forward,
        decide: (xs: Tensor) =>
          fromData(
            Int32Array.from(toFlat(probability(xs)), (p) => (p > 0.5 ? 1 : 0)),
            [(asRows(xs) as Tensor).shape[0]],
          ),
        predictive: (xs: Tensor) => Bernoulli(probability(xs)) as Univariate<Tensor>,
      }
      return withSampling(withExpectation(base))
    },
  }
}

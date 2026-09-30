/**
 * Logistic regression (binary and multinomial) fitted by Newton's method, i.e. iteratively reweighted least squares:
 * a reference estimator whose training loop is a traceable `Algorithm`.
 */

import { lstsq } from 'aifn/linalg'
import { fromData, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/trace'
import type { Decides, Expects, Fitted, Predicts, Samples, Scores, Trained } from './capabilities'
import type { Estimator, FitOptions, Supervised } from './data'
import {
  bernoulliPredictive,
  categoricalPredictive,
  type BernoulliPredictive,
  type CategoricalPredictive,
} from './distribution'
import { withExpectation, withSampling } from './mixins'
import { labels, matrixShape, values } from './util'

/** Hyperparameters of `logisticRegression`. */
export interface LogisticRegressionParams {
  /**
   * L2 penalty λ in Σᵢ −log p(yᵢ | xᵢ) + ½λ‖W‖² (intercepts are not penalised). scikit-learn's `C` is 1/λ. Default 1,
   * scikit-learn's default.
   */
  l2?: number
  /** Fit intercepts (default true). */
  intercept?: boolean
  /** Use the softmax model even for two classes (default: only for three or more). */
  multinomial?: boolean
  /** Stop when half the squared Newton decrement, an estimate of f − f*, is at most `tol` (default 1e-12). */
  tol?: number
  /** Most Newton steps (default 100). */
  maxIterations?: number
}

/** The problem an IRLS run solves: the design with a trailing column of ones when fitting intercepts. */
export interface IrlsProblem {
  /** Design matrix [n, p] (p = d + 1 with an intercept column last). */
  design: Tensor
  /** Class labels 0 … K−1, length n. */
  labels: Tensor
  /** Number of weight columns: 1 for the binary model, K for the multinomial one. */
  columns: number
  /** Penalty λ. */
  l2: number
  /** True when the last design column is the (unpenalised) intercept. */
  intercept: boolean
  tol: number
}

/** One IRLS state: the weights and everything computed at them. */
export interface IrlsState {
  /** Weights [p, C]: row j for design column j (the intercept last), column k for class k (C = 1 when binary). */
  weights: Tensor
  /** Penalised negative log-likelihood at `weights`. */
  loss: number
  /** Gradient of the loss, [p, C]. */
  gradient: Tensor
  /** Newton direction H⁻¹g (minimum norm when H is singular, as in the multinomial model's intercepts), [p, C]. */
  direction: Tensor
  /** Newton decrement λ(W) = √(gᵀH⁻¹g) (Boyd and Vandenberghe, 2004, §9.5.1). */
  decrement: number
  /** Step length taken to reach this state by backtracking (1 is the full Newton step; 0 at the start). */
  stepSize: number
  iteration: number
  /** λ²/2 ≤ tol. */
  converged: boolean
  /** The loss is not finite. */
  diverged: boolean
  /** The line search could not decrease the loss (rounding level reached). */
  stalled: boolean
}

type Evaluated = { loss: number; gradient: Float64Array; hessian: Float64Array | null }

/** Loss, gradient and (optionally) Hessian of the penalised negative log-likelihood at W (flattened [p, C]). */
function evaluate(problem: IrlsProblem, W: Float64Array, withHessian: boolean): Evaluated {
  const X = values(problem.design)
  const y = values(problem.labels)
  const [n, p] = problem.design.shape
  const C = problem.columns
  const P = p * C
  const gradient = new Float64Array(P)
  const hessian = withHessian ? new Float64Array(P * P) : null
  let loss = 0
  const eta = new Float64Array(C)
  const prob = new Float64Array(C)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < C; k++) {
      let s = 0
      for (let a = 0; a < p; a++) s += X[i * p + a] * W[a * C + k]
      eta[k] = s
    }
    const yi = y[i]
    if (C === 1) {
      // −log p(y | η) = softplus(η) − yη, computed stably; residual σ(η) − y and weight σ(1 − σ).
      const e = eta[0]
      loss += (e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e))) - yi * e
      prob[0] = e >= 0 ? 1 / (1 + Math.exp(-e)) : Math.exp(e) / (1 + Math.exp(e))
      const r = prob[0] - yi
      const w = prob[0] * (1 - prob[0])
      for (let a = 0; a < p; a++) {
        gradient[a] += X[i * p + a] * r
        if (hessian) for (let b = 0; b <= a; b++) hessian[a * P + b] += w * X[i * p + a] * X[i * p + b]
      }
    } else {
      let m = -Infinity
      for (let k = 0; k < C; k++) m = Math.max(m, eta[k])
      let z = 0
      for (let k = 0; k < C; k++) z += Math.exp(eta[k] - m)
      const lse = m + Math.log(z)
      loss += lse - eta[yi]
      for (let k = 0; k < C; k++) prob[k] = Math.exp(eta[k] - lse)
      for (let a = 0; a < p; a++) {
        const xa = X[i * p + a]
        for (let k = 0; k < C; k++) gradient[a * C + k] += xa * (prob[k] - (k === yi ? 1 : 0))
      }
      if (hessian) {
        // H[(a,k),(b,l)] = Σᵢ x_ia x_ib p_k (δ_kl − p_l).
        for (let a = 0; a < p; a++) {
          for (let k = 0; k < C; k++) {
            const row = (a * C + k) * P
            for (let b = 0; b <= a; b++) {
              const xx = X[i * p + a] * X[i * p + b]
              for (let l = 0; l < C; l++) {
                if (b * C + l > a * C + k) break
                hessian[row + b * C + l] += xx * prob[k] * ((k === l ? 1 : 0) - prob[l])
              }
            }
          }
        }
      }
    }
  }
  const penalised = problem.intercept ? p - 1 : p
  for (let a = 0; a < penalised; a++) {
    for (let k = 0; k < C; k++) {
      const j = a * C + k
      loss += 0.5 * problem.l2 * W[j] * W[j]
      gradient[j] += problem.l2 * W[j]
      if (hessian) hessian[j * P + j] += problem.l2
    }
  }
  if (hessian) for (let r = 0; r < P; r++) for (let c = r + 1; c < P; c++) hessian[r * P + c] = hessian[c * P + r]
  return { loss, gradient, hessian }
}

/** The state at W: loss, gradient, Newton direction and decrement. */
function stateAt(
  problem: IrlsProblem,
  W: Float64Array,
  iteration: number,
  stepSize: number,
  stalled = false,
): IrlsState {
  const [, p] = problem.design.shape
  const C = problem.columns
  const P = p * C
  const { loss, gradient, hessian } = evaluate(problem, W, true)
  const shape = [p, C]
  let direction = new Float64Array(P)
  let decrement = NaN
  if (Number.isFinite(loss)) {
    direction = values(lstsq(fromData(hessian!, [P, P]), fromData(gradient, [P])).x).slice()
    let gd = 0
    for (let j = 0; j < P; j++) gd += gradient[j] * direction[j]
    decrement = Math.sqrt(Math.max(gd, 0))
  }
  return {
    weights: fromData(W, shape),
    loss,
    gradient: fromData(gradient, shape),
    direction: fromData(direction, shape),
    decrement,
    stepSize,
    iteration,
    converged: (decrement * decrement) / 2 <= problem.tol,
    diverged: !Number.isFinite(loss),
    stalled,
  }
}

/**
 * Newton's method for (penalised) logistic regression as a traceable algorithm. For the binary model the Newton step
 * is weighted least squares with weights σ(1 − σ) (IRLS; Nelder and Wedderburn, 1972; Hastie, Tibshirani and
 * Friedman, 2009, §4.4.1). Each step backtracks from the full Newton step until the Armijo condition
 * f(W − tΔ) ≤ f(W) − 10⁻⁴·t·λ² holds (Boyd and Vandenberghe, 2004, Algorithm 9.5), which makes the method globally
 * convergent. `init` takes starting weights [p, C] (default zeros).
 */
export function logisticIrls(problem: IrlsProblem): Algorithm<{ weights?: Tensor }, IrlsState> {
  return {
    name: 'logistic-irls',
    init: ({ weights }) => {
      const [, p] = problem.design.shape
      const W = weights ? values(weights).slice() : new Float64Array(p * problem.columns)
      return stateAt(problem, W, 0, 0)
    },
    step: (state) => {
      const W = values(state.weights)
      const d = values(state.direction)
      const slope = state.decrement * state.decrement
      let t = 1
      while (t > 1e-12) {
        const next = Float64Array.from(W, (w, j) => w - t * d[j])
        const { loss } = evaluate(problem, next, false)
        if (loss <= state.loss - 1e-4 * t * slope) return stateAt(problem, next, state.iteration + 1, t)
        t /= 2
      }
      return { ...state, iteration: state.iteration + 1, stepSize: 0, stalled: true }
    },
    done: (state) => state.converged || state.stalled,
  }
}

/** A fitted logistic regression. */
export interface LogisticRegressionModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, BernoulliPredictive | CategoricalPredictive>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Scores<Tensor>,
    Trained<IrlsState> {
  readonly kind: 'logistic-regression'
  /** Number of classes K (labels 0 … K−1). */
  readonly classes: number
  /** True for the softmax model with one weight column per class. */
  readonly multinomial: boolean
  /** Coefficients: [d] for the binary model, [d, K] for the multinomial one. */
  readonly weights: Tensor
  /** Intercepts: a scalar tensor [] for the binary model, [K] for the multinomial one (all 0 without intercepts). */
  readonly intercept: Tensor
  /** The final penalised negative log-likelihood. */
  readonly loss: number
  readonly converged: boolean
  readonly iterations: number
  readonly l2: number
}

/**
 * Logistic regression for labels 0 … K−1: Bernoulli with P(y = 1 | x) = σ(x·w + b) for two classes, softmax over K
 * weight columns otherwise, with an L2 penalty on the weights. Fitted by Newton's method (`logisticIrls`); the run is
 * kept in `training`. The multinomial Hessian is singular along "add a constant to every intercept"; the minimum-norm
 * Newton step keeps Σₖ bₖ = 0, as scikit-learn's solution has.
 *
 * Capabilities: `forward` and `score` (logits: [N] binary, [N, K] multinomial), `decide` (the most probable class;
 * logit > 0 when binary), `predictive` (Bernoulli or categorical), `expect` (P(y = 1), or E[k]), `sample`.
 */
export function logisticRegression(
  params: LogisticRegressionParams = {},
): Estimator<Supervised<Tensor, Tensor>, LogisticRegressionModel> {
  const { l2 = 1, intercept = true, multinomial = false, tol = 1e-12, maxIterations = 100 } = params
  if (!(l2 >= 0)) throw new Error('logisticRegression: l2 must be non-negative')
  return {
    name: 'logistic-regression',
    params: { l2, intercept, multinomial, tol, maxIterations },
    fit({ x, y }, options: FitOptions = {}) {
      const [n, d] = matrixShape(x, 'logisticRegression')
      const target = labels(y, 'logisticRegression')
      if (target.length !== n) throw new Error(`logisticRegression: ${n} rows of x but ${target.length} labels`)
      let K = 0
      for (const v of target) {
        if (!(Number.isInteger(v) && v >= 0)) throw new Error('logisticRegression: labels must be integers 0 … K−1')
        K = Math.max(K, v + 1)
      }
      K = Math.max(K, 2)
      const softmax = multinomial || K > 2
      const C = softmax ? K : 1
      const p = intercept ? d + 1 : d
      const X = values(x)
      const design = new Float64Array(n * p)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) design[i * p + j] = X[i * d + j]
        if (intercept) design[i * p + d] = 1
      }
      const problem: IrlsProblem = {
        design: fromData(design, [n, p]),
        labels: fromData(Float64Array.from(target), [n]),
        columns: C,
        l2,
        intercept,
        tol,
      }
      const training: Trace<IrlsState> = trace(logisticIrls(problem), {}, maxIterations, {
        every: options.trace?.every ?? 1,
        checkpointEvery: options.trace?.checkpointEvery,
        record: {
          loss: (s) => s.loss,
          decrement: (s) => s.decrement,
          stepSize: (s) => s.stepSize,
          ...(options.trace?.record as Record<string, (s: IrlsState, t: number) => number> | undefined),
        },
      })
      const final = training.steps[training.steps.length - 1]
      const W = values(final.weights)
      const coef = new Float64Array(d * C)
      const bias = new Float64Array(C)
      for (let j = 0; j < d; j++) for (let k = 0; k < C; k++) coef[j * C + k] = W[j * C + k]
      if (intercept) for (let k = 0; k < C; k++) bias[k] = W[d * C + k]
      const forward = (input: Tensor): Tensor => {
        const [m, cols] = matrixShape(input, 'logisticRegression.forward')
        if (cols !== d) throw new Error(`logisticRegression: fitted on ${d} features, given ${cols}`)
        const Z = values(input)
        const out = new Float64Array(m * C)
        for (let i = 0; i < m; i++) {
          for (let k = 0; k < C; k++) {
            let s = bias[k]
            for (let j = 0; j < d; j++) s += Z[i * d + j] * coef[j * C + k]
            out[i * C + k] = s
          }
        }
        return fromData(out, C === 1 ? [m] : [m, C])
      }
      const predictive = (input: Tensor): BernoulliPredictive | CategoricalPredictive => {
        const eta = values(forward(input))
        if (C === 1) {
          return bernoulliPredictive(
            fromData(
              Float64Array.from(eta, (e) => (e >= 0 ? 1 / (1 + Math.exp(-e)) : Math.exp(e) / (1 + Math.exp(e)))),
              [eta.length],
            ),
          )
        }
        const m = eta.length / C
        const probs = new Float64Array(eta.length)
        for (let i = 0; i < m; i++) {
          let mx = -Infinity
          for (let k = 0; k < C; k++) mx = Math.max(mx, eta[i * C + k])
          let z = 0
          for (let k = 0; k < C; k++) z += probs[i * C + k] = Math.exp(eta[i * C + k] - mx)
          for (let k = 0; k < C; k++) probs[i * C + k] /= z
        }
        return categoricalPredictive(fromData(probs, [m, C]))
      }
      const decide = (input: Tensor): Tensor => {
        const eta = values(forward(input))
        const m = eta.length / C
        const out = new Int32Array(m)
        for (let i = 0; i < m; i++) {
          if (C === 1) out[i] = eta[i] > 0 ? 1 : 0
          else for (let k = 1; k < C; k++) if (eta[i * C + k] > eta[i * C + out[i]]) out[i] = k
        }
        return fromData(out, [m])
      }
      const base = {
        kind: 'logistic-regression' as const,
        classes: K,
        multinomial: softmax,
        weights: fromData(coef, C === 1 ? [d] : [d, C]),
        intercept: fromData(bias, C === 1 ? [] : [C]),
        loss: final.loss,
        converged: final.converged,
        iterations: final.iteration,
        l2,
        training,
        forward,
        score: forward,
        decide,
        predictive,
      }
      return withSampling(withExpectation(base))
    },
  }
}

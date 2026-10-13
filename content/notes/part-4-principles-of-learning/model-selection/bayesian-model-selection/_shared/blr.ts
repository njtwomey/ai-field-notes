/**
 * Shared problems for the Bayesian model selection figures. Everything composes engine calls: the data come from
 * `regression1d`, the models are Bayesian polynomial regressions scored by `bayesianLinearEvidence`, the posterior
 * draws come from `MultivariateNormal`, and the predictive criteria are `waic`, `psisLoo` and `dic`.
 */
import { stream, type Stream } from 'aifn-compute/foundation/random'
import { add, matmul, mean, mul, sqrt, sum, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { MultivariateNormal, Normal } from 'aifn-compute/probability/distributions'
import { regression1d } from 'aifn-methods/data/synthetic'
import { bayesianLinearEvidence, type LinearEvidence } from 'aifn-methods/learning/linear'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { targetFunction } from 'aifn-methods/theory'

/** The regression functions on $[-1, 1]$: $\sin(\pi x)$, which no polynomial equals, and the cubic $x^3 - x$. */
export const TARGETS = {
  sine: targetFunction('sine'),
  cubic: 'cubic' as const,
}
export type Target = keyof typeof TARGETS

/** $n$ points of $y = f(x) + \sigma\varepsilon$ with $x$ uniform on $[-1, 1]$; `x` is $n \times 1$. */
export function drawData(s: Stream | string, n: number, noise: number, target: Target = 'sine') {
  const d = regression1d(typeof s === 'string' ? stream(s) : s, { n, fn: TARGETS[target], noise, range: [-1, 1] })
  return { x: d.x as Tensor, y: d.y as Tensor }
}

/** The polynomial design $[1, x, \dots, x^d]$ (with `bias`, the default) of the inputs, fitted on `x` itself. */
export function design(x: Tensor, degree: number, bias = true) {
  const pf = polynomialFeatures({ degree, includeBias: bias }).fit({ x })
  return { X: pf.transform(x) as Tensor, map: (z: Tensor) => pf.transform(z) as Tensor }
}

/** The pointwise log-likelihood $\log p(y_i \mid \wvec^{(s)})$ of $S$ posterior draws, $S \times n$. */
export function logLikDraws(ev: LinearEvidence, X: Tensor, y: Tensor, draws: number, seed: string) {
  const W = MultivariateNormal(ev.mean, { covariance: ev.covariance }).sample(stream(seed), {
    shape: [draws],
  }) as Tensor
  const pred = matmul(W, transpose(X))
  return Normal(pred, 1 / Math.sqrt(ev.beta)).logProb(y) as Tensor
}

/** The posterior mean $\Phimat\mvec_N$ of the regression function at the rows of a design. */
export function posteriorMean(ev: LinearEvidence, X: Tensor) {
  return matmul(X, ev.mean) as Tensor
}

/** The pointwise log-likelihood at the posterior mean, $n$ values. */
export function logLikAtMean(ev: LinearEvidence, X: Tensor, y: Tensor) {
  return Normal(matmul(X, ev.mean), 1 / Math.sqrt(ev.beta)).logProb(y) as Tensor
}

/**
 * The log density the posterior predictive $\Gauss(\phivec^\top\mvec_N, \beta^{-1} + \phivec^\top\Amat^{-1}\phivec)$
 * gives each point of a sample, $m$ values.
 */
export function predictiveLogDensity(ev: LinearEvidence, Xnew: Tensor, ynew: Tensor) {
  const m = matmul(Xnew, ev.mean)
  const v = sum(mul(matmul(Xnew, ev.covariance), Xnew), 1) as Tensor
  return Normal(m, sqrt(add(v, 1 / ev.beta))).logProb(ynew) as Tensor
}

/**
 * The true elpd of a fitted model on $n$ new points: $n$ times the mean log predictive density on a large fresh sample
 * from the same generator.
 */
export function trueElpd(ev: LinearEvidence, Xfresh: Tensor, yfresh: Tensor, n: number) {
  return n * Number(mean(predictiveLogDensity(ev, Xfresh, yfresh)))
}

/** The evidence of a polynomial model of degree $d$ at prior precision $\alpha$ and noise precision $\beta$. */
export function polynomialEvidence(x: Tensor, y: Tensor, degree: number, alpha: number, beta: number) {
  const { X, map } = design(x, degree)
  return { X, map, ev: bayesianLinearEvidence(X, y, { alpha, beta }) }
}

export const flat = (t: unknown) => Array.from(toFlat(t as Tensor))
export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? Number(v.toPrecision(digits)).toString() : '—')

/**
 * Generalised linear models as estimators: IRLS fits with the classical inference of McCullagh and Nelder (1989)
 * (standard errors from the inverse Fisher information, Wald tests, deviance, Pearson dispersion, AIC, residuals), and
 * the negative binomial with its shape estimated by maximum likelihood (Venables and Ripley, 2002, §7.4, `glm.nb`).
 */

import {
  withExpectation,
  withSampling,
  type Decides,
  type Distribution,
  type Estimator,
  type Expects,
  type FitOptions,
  type Fitted,
  type Predicts,
  type Samples,
  type Trained,
} from 'aifn/estimators'
import { cholesky, inverse, pinv } from 'aifn/linalg'
import { findRoot } from 'aifn/solve'
import { digamma, normalCdf, studentTCdf } from 'aifn/special'
import { fromData, toFlat, type Tensor } from 'aifn/tensor'
import { trace, type Trace } from 'aifn/trace'
import { link as linkByName, negativeBinomial, type Family, type Link, type LinkName } from './families'
import { irls, type IrlsState } from './irls'

/** Data for a GLM fit: inputs x [n, d], responses y [n], optional prior weights and offset [n]. */
export type GlmData = {
  x: Tensor
  y: Tensor
  /** Prior weights (binomial: the number of trials, with y the proportion of successes). */
  weights?: Tensor
  /** Offset added to the linear predictor, e.g. log exposure in a Poisson rate model. */
  offset?: Tensor
}

/** Hyperparameters of `glm`. */
export type GlmParams = {
  family: Family
  /** The link (default: the family's default link). */
  link?: LinkName | Link
  /** Fit an intercept (default true); it is the last coefficient. */
  intercept?: boolean
  /** Ridge penalty λ on ½‖β‖²… added as λI to XᵀWX, intercept unpenalised (default 0). */
  l2?: number
  tol?: number
  /** Most IRLS steps (default 50). */
  maxIterations?: number
}

/** The kinds of residual (McCullagh and Nelder, 1989, §2.4). */
export type ResidualKind = 'response' | 'pearson' | 'deviance' | 'working'

/** A fitted GLM. */
export interface GlmModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Distribution>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<IrlsState> {
  readonly kind: 'glm'
  readonly family: Family
  readonly link: Link
  /** Coefficients β [p]: one per column of x, then the intercept (when fitted). */
  readonly coefficients: Tensor
  /** Names "x0", "x1", …, "intercept". */
  readonly names: string[]
  /** Covariance φ(XᵀWX + λI)⁻¹ of β̂ [p, p] (Bayesian for a penalised fit). */
  readonly covariance: Tensor
  /** √diag(covariance) [p]. */
  readonly standardErrors: Tensor
  /** Wald statistics β̂ⱼ/SEⱼ [p]: z when φ is known, t (n − p degrees of freedom) when it is estimated. */
  readonly statistics: Tensor
  /** Two-sided p-values of the Wald tests [p]. */
  readonly pValues: Tensor
  /** φ: fixed by the family, or the Pearson estimate Σ wᵢ(yᵢ − μᵢ)²/V(μᵢ) / (n − p). */
  readonly dispersion: number
  readonly deviance: number
  /** Deviance of the intercept-only model (with the same offset and weights). */
  readonly nullDeviance: number
  /** Residual degrees of freedom n − p (n − edf when penalised). */
  readonly dfResidual: number
  /** Effective degrees of freedom tr((XᵀWX + λI)⁻¹XᵀWX) (p when unpenalised). */
  readonly edf: number
  /** Log-likelihood at β̂, with φ = deviance/n for families that estimate it (R's convention for AIC). */
  readonly logLikelihood: number
  /** −2 log L + 2k, with k = p (+1 when φ is estimated). */
  readonly aic: number
  /** Fitted means μ̂ [n] and linear predictor η̂ [n] on the training data. */
  readonly fitted: Tensor
  readonly linearPredictor: Tensor
  readonly converged: boolean
  readonly iterations: number
  /** Training residuals of a kind. */
  residuals(kind?: ResidualKind): Tensor
  /** The linear predictor at new inputs, with an optional offset. */
  forward(x: Tensor, offset?: Tensor): Tensor
}

const vec = (a: Float64Array) => fromData(a, [a.length])
const flat = (t: Tensor) => Float64Array.from(toFlat(t))

function designOf(x: Tensor, intercept: boolean): { X: Float64Array; n: number; d: number; p: number } {
  if (x.shape.length !== 2) throw new Error(`glm: x must be [n, d], got [${x.shape.join(', ')}]`)
  const [n, d] = x.shape
  const p = d + (intercept ? 1 : 0)
  const v = toFlat(x)
  const X = new Float64Array(n * p)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X[i * p + j] = v[i * d + j]
    if (intercept) X[i * p + d] = 1
  }
  return { X, n, d, p }
}

/**
 * A generalised linear model g(E[y | x]) = xᵀβ + o, fitted by IRLS (`irls`), with the classical Wald inference.
 * Capabilities: `forward` (η), `decide` and `expect` (μ = g⁻¹(η)), `predictive` (the family's distribution at μ with
 * the fitted dispersion), `sample`. The IRLS run is kept in `training`.
 *
 * @example glm({ family: poisson() }).fit({ x, y, offset: logExposure })
 */
export function glm(params: GlmParams): Estimator<GlmData, GlmModel> {
  const { family, intercept = true, l2 = 0, tol = 1e-8, maxIterations = 50 } = params
  const link = typeof params.link === 'object' ? params.link : linkByName(params.link ?? family.defaultLink)
  return {
    name: `glm-${family.name}-${link.name}`,
    params,
    fit(data, options: FitOptions = {}) {
      const { X, n, d, p } = designOf(data.x, intercept)
      const design = fromData(X, [n, p])
      let penalty: Tensor | undefined
      if (l2 > 0) {
        const P = new Float64Array(p * p)
        for (let j = 0; j < d; j++) P[j * p + j] = l2
        if (!intercept) P[(p - 1) * p + p - 1] = l2
        penalty = fromData(P, [p, p])
      }
      const problem = { design, y: data.y, family, link, weights: data.weights, offset: data.offset, penalty, tol }
      const training: Trace<IrlsState> = trace(irls(problem), {}, maxIterations, {
        every: options.trace?.every ?? 1,
        record: { deviance: (s) => s.deviance, penalisedDeviance: (s) => s.penalisedDeviance },
      })
      const final = training.steps[training.steps.length - 1]
      if (!final.coefficients) throw new Error('glm: IRLS took no step')
      return summarise({ family, link, intercept, d, n, p, X, data, final, training, penalty })
    },
  }
}

type SummaryInput = {
  family: Family
  link: Link
  intercept: boolean
  d: number
  n: number
  p: number
  X: Float64Array
  data: GlmData
  final: IrlsState
  training: Trace<IrlsState>
  penalty?: Tensor
}

function summarise(s: SummaryInput): GlmModel {
  const { family, link, intercept, d, n, p, X, data, final, training, penalty } = s
  const beta = flat(final.coefficients!)
  const mu = flat(final.mu)
  const eta = flat(final.eta)
  const y = flat(data.y)
  const w = data.weights ? flat(data.weights) : new Float64Array(n).fill(1)
  // Working weights at the final β (the state's W is computed at its η).
  const W = flat(final.workingWeights)
  const I = new Float64Array(p * p)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) I[a * p + b] += W[i] * X[i * p + a] * X[i * p + b]
  const H = Float64Array.from(I)
  if (penalty) toFlat(penalty).forEach((v, k) => (H[k] += v))
  const factor = cholesky(fromData(H, [p, p]), { jitter: false })
  const Hinv = flat(factor.failed ? pinv(fromData(H, [p, p])) : (inverse(fromData(H, [p, p])) as Tensor))
  let edf = 0
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) edf += Hinv[a * p + b] * I[b * p + a]
  const dfResidual = n - edf
  const V = flat(family.variance(vec(mu)) as Tensor)
  let pearson = 0
  for (let i = 0; i < n; i++) pearson += (w[i] * (y[i] - mu[i]) ** 2) / V[i]
  const dispersion = family.dispersion ?? pearson / dfResidual
  const covariance = Hinv.map((v) => v * dispersion)
  const se = Float64Array.from({ length: p }, (_, j) => Math.sqrt(covariance[j * p + j]))
  const statistics = Float64Array.from(beta, (b, j) => b / se[j])
  const pValues = Float64Array.from(statistics, (t) =>
    family.dispersion === null
      ? 2 * (studentTCdf(-Math.abs(t), dfResidual) as number)
      : 2 * (normalCdf(-Math.abs(t)) as number),
  )
  const unit = flat(family.unitDeviance(data.y, final.mu) as Tensor)
  const devianceResiduals = Float64Array.from(
    unit,
    (u, i) => Math.sign(y[i] - mu[i]) * Math.sqrt(Math.max(w[i] * u, 0)),
  )
  // The null model: intercept only (or the offset alone), same family, link, weights and offset.
  let nullDeviance: number
  if (intercept) {
    const nullRun = trace(
      irls({
        design: fromData(new Float64Array(n).fill(1), [n, 1]),
        y: data.y,
        family,
        link,
        weights: data.weights,
        offset: data.offset,
      }),
      {},
      50,
    )
    nullDeviance = nullRun.steps[nullRun.steps.length - 1].deviance
  } else {
    const o = data.offset ?? fromData(new Float64Array(n), [n])
    const muNull = link.inverse(o) as Tensor
    const u = flat(family.unitDeviance(data.y, muNull) as Tensor)
    nullDeviance = u.reduce((a, v, i) => a + w[i] * v, 0)
  }
  const phiLik = family.dispersion ?? final.deviance / n
  const logLikelihood = family.logLikelihood(data.y, final.mu, phiLik, data.weights ?? vec(w))
  const k = p + (family.dispersion === null ? 1 : 0)
  const names = [...Array.from({ length: d }, (_, j) => `x${j}`), ...(intercept ? ['intercept'] : [])]

  const forward = (x: Tensor, offset?: Tensor) => {
    const { X: Z, n: m, d: cols } = designOf(x, intercept)
    if (cols !== d) throw new Error(`glm: fitted on ${d} features, given ${cols}`)
    const o = offset ? flat(offset) : null
    const out = new Float64Array(m)
    for (let i = 0; i < m; i++) {
      let t = o ? o[i] : 0
      for (let a = 0; a < p; a++) t += Z[i * p + a] * beta[a]
      out[i] = t
    }
    return vec(out)
  }
  const meanAt = (x: Tensor) => link.inverse(forward(x)) as Tensor
  const base = {
    kind: 'glm' as const,
    family,
    link,
    coefficients: vec(beta),
    names,
    covariance: fromData(covariance, [p, p]),
    standardErrors: vec(se),
    statistics: vec(statistics),
    pValues: vec(pValues),
    dispersion,
    deviance: final.deviance,
    nullDeviance,
    dfResidual,
    edf,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * k,
    fitted: final.mu,
    linearPredictor: final.eta,
    converged: final.converged,
    iterations: final.iteration,
    training,
    residuals: (kind: ResidualKind = 'deviance') => {
      if (kind === 'response') return vec(Float64Array.from(y, (v, i) => v - mu[i]))
      if (kind === 'pearson')
        return vec(Float64Array.from(y, (v, i) => ((v - mu[i]) * Math.sqrt(w[i])) / Math.sqrt(V[i])))
      if (kind === 'working') {
        const dmu = flat(link.derivative(vec(eta)) as Tensor)
        return vec(Float64Array.from(y, (v, i) => (v - mu[i]) / dmu[i]))
      }
      return vec(devianceResiduals)
    },
    forward,
    decide: meanAt,
    predictive: (x: Tensor) => family.predictive(meanAt(x), dispersion),
  }
  return withSampling(withExpectation(base)) as GlmModel
}

// ── Negative binomial with θ estimated ───────────────────────────────────────────────────────────────────────────

/** Hyperparameters of `negativeBinomialRegression`. */
export type NegativeBinomialParams = Omit<GlmParams, 'family'> & {
  /** Starting θ (default: the moment estimate from a Poisson fit). */
  theta?: number
  /** Most alternations between β and θ (default 25). */
  maxAlternations?: number
}

/** The ML estimate of θ given fitted means (the score equation, solved by Brent's method in log θ). */
export function thetaMaximumLikelihood(y: Tensor, mu: Tensor, weights?: Tensor): { theta: number; converged: boolean } {
  const ys = flat(y)
  const ms = flat(mu)
  const w = weights ? flat(weights) : new Float64Array(ys.length).fill(1)
  const score = (logTheta: number) => {
    const t = Math.exp(logTheta)
    let s = 0
    for (let i = 0; i < ys.length; i++)
      s +=
        w[i] *
        ((digamma(ys[i] + t) as number) -
          (digamma(t) as number) +
          Math.log(t) +
          1 -
          Math.log(t + ms[i]) -
          (ys[i] + t) / (ms[i] + t))
    return s
  }
  const r = findRoot(score, [Math.log(1e-4), Math.log(1e6)])
  return { theta: Math.exp(r.x), converged: r.converged }
}

/**
 * Negative binomial (NB2) regression with θ estimated: alternate an IRLS fit of β at fixed θ with the ML estimate of
 * θ at fixed μ until θ settles (Venables and Ripley, 2002, §7.4). Returns the final GLM (with `family.params.theta`)
 * and the θ path.
 */
export function negativeBinomialRegression(
  params: NegativeBinomialParams = {},
): Estimator<GlmData, GlmModel & { thetaPath: number[] }> {
  const { maxAlternations = 25, ...rest } = params
  return {
    name: 'negative-binomial-regression',
    params,
    fit(data, options) {
      let theta = params.theta ?? 1
      const path = [theta]
      let model = glm({ ...rest, family: negativeBinomial(theta) }).fit(data, options)
      for (let k = 0; k < maxAlternations; k++) {
        const next = thetaMaximumLikelihood(data.y, model.fitted, data.weights).theta
        path.push(next)
        const change = Math.abs(Math.log(next / theta))
        theta = next
        model = glm({ ...rest, family: negativeBinomial(theta) }).fit(data, options)
        if (change < 1e-8) break
      }
      return { ...model, thetaPath: path }
    },
  }
}

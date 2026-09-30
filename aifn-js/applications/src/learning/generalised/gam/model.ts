/**
 * Generalised additive models g(E[y]) = α + Σⱼ fⱼ(x) fitted by penalised IRLS (`aifn-applied/learning/glm`'s `irls`
 * with the penalty S_λ = Σ λₖ Sₖ), with smoothing parameters chosen by GCV/UBRE or REML (Laplace-approximate restricted
 * likelihood), effective degrees of freedom, Bayesian posterior bands and draws, and shape constraints.
 *
 * References: Wood (2017), "Generalized Additive Models: An Introduction with R", 2nd ed.: penalised IRLS §6.1.1,
 * GCV and UBRE §6.2.3–6.2.4, REML and LAML §6.2.5–6.2.6, EDF §6.1.2, posterior covariance V_β = (XᵀWX + S_λ)⁻¹φ §6.10.
 * Shape constraints follow pyGAM (Servén and Brummitt, 2018): a large penalty on the coefficient differences that
 * violate the constraint, added until none do.
 */

import {
  withExpectation,
  withSampling,
  type Decides,
  type Distribution,
  type Estimator,
  type Expects,
  type Fitted,
  type Predicts,
  type Samples,
  type Trained,
} from 'aifn/learning/estimators'
import { gaussianFamily, link as linkByName, type Family, type Link, type LinkName } from 'aifn/probability/likelihoods'
import type { IrlsState } from '../irls'
import { residuals, type ResidualKind } from '../residuals'
import {
  nullSpaceDimension,
  penalisedFit as fitAt,
  penaltyMatrix,
  smoothingCriterion as criterion,
  type PenalisedDesign,
} from '../smoothing'
import { cholesky } from 'aifn/numerics/linalg'
import { minimize } from 'aifn/optim/minimize'
import { normals, type Stream } from 'aifn/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { buildTerms, times, type BuiltTerm, type TermSpec } from './terms'

type F64 = Float64Array
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))
const vec = (a: F64) => fromData(a, [a.length])

/**
 * Data for a GAM: features x [n, d] (factor columns as integer codes), responses y [n], optional weights and offset.
 */
export type GamData = { x: Tensor; y: Tensor; weights?: Tensor; offset?: Tensor }

/** How smoothing parameters are chosen. */
export type SmoothingMethod = 'reml' | 'gcv' | 'fixed'

/** Hyperparameters of `gam`. */
export type GamParams = {
  terms: readonly TermSpec[]
  family?: Family
  link?: LinkName | Link
  /** `reml` (default), `gcv` (GCV when φ is estimated, UBRE when it is known) or `fixed` (λ = 1 unless given). */
  method?: SmoothingMethod
  /** GCV/UBRE inflation γ ≥ 1 of the EDF, against overfitting (default 1). */
  gamma?: number
  /** Most P-IRLS steps per fit (default 50). */
  maxSteps?: number
  /** Most Nelder–Mead steps of the smoothing-parameter search (default 200). */
  maxSearchSteps?: number
  /** Weight of the shape-constraint penalty relative to the mean diagonal of XᵀWX (default 1e6). */
  constraintWeight?: number
}

/** The model matrix, penalties and fixed data of a GAM (a `PenalisedDesign` with its terms). */
type Assembled = PenalisedDesign & {
  terms: BuiltTerm[]
  /** Column offset of each term's block (column 0 is the intercept). */
  offsets: number[]
  P: number
  n: number
  d: number
  X: F64
  /** Each penalty embedded in P × P, with the term it belongs to. */
  penalties: { S: F64; term: number }[]
  /** Fixed λ per penalty (NaN: selected). */
  fixed: number[]
  /** Dimension of the unpenalised space (intercept and null spaces). */
  nullSpace: number
}

function assemble(specs: readonly TermSpec[], x: Tensor): Assembled {
  if (x.shape.length !== 2) throw new Error('gam: x must be [n, d]')
  const [n, d] = x.shape
  const Xin = f64(x)
  const terms = buildTerms(specs, Xin, n, d)
  const offsets: number[] = []
  let P = 1
  for (const t of terms) {
    offsets.push(P)
    P += t.size
  }
  const X = new Float64Array(n * P)
  for (let i = 0; i < n; i++) X[i * P] = 1
  terms.forEach((t, j) => {
    const B = times(t.raw(Xin, n, d), n, t.rawSize, t.Z, t.size)
    for (let i = 0; i < n; i++) for (let c = 0; c < t.size; c++) X[i * P + offsets[j] + c] = B[i * t.size + c]
  })
  const penalties: { S: F64; term: number }[] = []
  const fixed: number[] = []
  terms.forEach((t, j) => {
    t.penalties.forEach((S, k) => {
      const E = new Float64Array(P * P)
      for (let a = 0; a < t.size; a++)
        for (let b = 0; b < t.size; b++) E[(offsets[j] + a) * P + offsets[j] + b] = S[a * t.size + b]
      penalties.push({ S: E, term: j })
      fixed.push(t.fixedLambda[k])
    })
  })
  return { terms, offsets, P, n, d, X, penalties, fixed, nullSpace: nullSpaceDimension(penalties, P) }
}

/** Partial effect of one term on a grid, with pointwise standard errors. */
export type PartialEffect = { fit: Tensor; se: Tensor }

/** A fitted GAM. */
export interface GamModel
  extends
    Fitted<Tensor, Tensor>,
    Decides<Tensor, Tensor>,
    Predicts<Tensor, Distribution>,
    Expects<Tensor>,
    Samples<Tensor, Tensor>,
    Trained<IrlsState> {
  readonly kind: 'model'
  /** The model's name. */
  readonly name: 'gam'
  readonly family: Family
  readonly link: Link
  readonly terms: readonly BuiltTerm[]
  /** Term labels, e.g. "s(x0)", "te(x1, x2)". */
  readonly labels: string[]
  /** All coefficients [P]: the intercept, then each term's block. */
  readonly coefficients: Tensor
  readonly intercept: number
  /** One smoothing parameter per penalty, in term order. */
  readonly lambdas: number[]
  /** Total effective degrees of freedom tr(H⁻¹XᵀWX), including the intercept. */
  readonly edf: number
  /** EDF of each term. */
  readonly termEdf: number[]
  /** φ: fixed by the family or estimated as Σwᵢ(yᵢ − μᵢ)²/V(μᵢ) / (n − edf). */
  readonly dispersion: number
  readonly deviance: number
  /** The smoothing-parameter criterion at the chosen λ (REML: −2 × restricted log-likelihood up to a constant). */
  readonly smoothingScore: { method: SmoothingMethod; value: number; evaluations: number }
  /** Bayesian covariance V_β = H⁻¹φ [P, P]. */
  readonly covariance: Tensor
  readonly fitted: Tensor
  readonly linearPredictor: Tensor
  readonly converged: boolean
  /** Jitter added to XᵀWX + S_λ to factor it (non-zero when unpenalised directions overlap). */
  readonly jitter: number
  /** Shape constraints: how many refits they took and how many differences still violate (with tolerance 1e-6). */
  readonly shape: { iterations: number; violations: number }
  /**
   * The partial effect fⱼ on a grid: [m] for a one-dimensional term (a `by` variable set to 1 or its level), [m, 2] for
   * a tensor.
   */
  partial(term: number, grid: Tensor): PartialEffect
  /** Partial residuals of term j at the training data: f̂ⱼ(xᵢ) + the working residual (yᵢ − μᵢ)/μ′(ηᵢ), [n]. */
  partialResiduals(term: number): Tensor
  /** Training residuals of a kind (default deviance). */
  residuals(kind?: ResidualKind): Tensor
  /** `count` posterior draws of fⱼ on a grid from β ~ N(β̂, V_β), [count, m]. */
  partialDraws(s: Stream, term: number, grid: Tensor, count: number): Tensor
}

/**
 * A generalised additive model: `gam({ terms: [s(0), s(1, { k: 20 }), linearTerm(2)], family: poisson() })`.
 * Capabilities: `forward` (η), `decide` and `expect` (μ), `predictive` (the family at μ with the fitted dispersion),
 * `sample`. The final P-IRLS run is kept in `training`.
 */
export function gam(params: GamParams): Estimator<GamData, GamModel> {
  const { terms: specs, family = gaussianFamily(), method = 'reml', gamma = 1, maxSteps = 50 } = params
  const { maxSearchSteps = 200, constraintWeight = 1e6 } = params
  const lk = typeof params.link === 'object' ? params.link : linkByName(params.link ?? family.defaultLink)
  return {
    name: `gam-${family.name}`,
    params,
    fit(data) {
      const A = assemble(specs, data.x)
      const free = A.fixed.map((v, k) => (Number.isNaN(v) ? k : -1)).filter((k) => k >= 0)
      const lambdasFrom = (logs: ArrayLike<number>) =>
        A.fixed.map((v, k) => (Number.isNaN(v) ? (method === 'fixed' ? 1 : Math.exp(logs[free.indexOf(k)])) : v))
      let lambdas = lambdasFrom(new Float64Array(free.length))
      let evaluations = 0
      if (method !== 'fixed' && free.length > 0) {
        let warm: F64 | undefined
        const objective = (logs: Tensor) => {
          evaluations++
          const l = toFlat(logs) as number[]
          // Keep log λ in [−15, 15]; beyond it the criterion is flat and the search wanders.
          const excess = l.reduce((s, v) => s + Math.max(0, Math.abs(v) - 15) ** 2, 0)
          const S = penaltyMatrix(A, lambdasFrom(l.map((v) => Math.max(-15, Math.min(15, v)))))
          try {
            const fit = fitAt(A, data, family, lk, S, maxSteps, warm)
            warm = fit.beta
            const v = criterion(method, A, family, fit, S, gamma)
            return Number.isFinite(v) ? v + excess : Infinity
          } catch {
            return Infinity
          }
        }
        const x0 = new Float64Array(free.length)
        const simplex = [Array.from(x0), ...free.map((_, i) => Array.from(x0, (v, j) => v + (i === j ? 2 : 0)))]
        const best = minimize(objective, x0, {
          method: 'nelder-mead',
          maxSteps: maxSearchSteps,
          initialSimplex: simplex,
          xTolerance: 1e-4,
          fTolerance: 1e-8,
        })
        lambdas = lambdasFrom(Array.from(toFlat(best.x)).map((v) => Math.max(-15, Math.min(15, v))))
      }
      const S = penaltyMatrix(A, lambdas)
      let fit = fitAt(A, data, family, lk, S, maxSteps)
      const score = method === 'fixed' ? NaN : criterion(method, A, family, fit, S, gamma)

      // Shape constraints: penalise violated differences heavily, adding rows until none are violated.
      let shapeIterations = 0
      const active = A.terms.map(() => new Set<number>())
      const shapeRows = A.terms.map((t) =>
        t.shape ? times(t.shape.matrix, t.shape.rows, t.rawSize, t.Z, t.size) : null,
      )
      const violationsOf = (beta: F64, tol: number) => {
        const out: [number, number][] = []
        A.terms.forEach((t, j) => {
          const R = shapeRows[j]
          if (!R || !t.shape) return
          for (let r = 0; r < t.shape.rows; r++) {
            let v = 0
            for (let c = 0; c < t.size; c++) v += R[r * t.size + c] * beta[A.offsets[j] + c]
            if (v < -tol) out.push([j, r])
          }
        })
        return out
      }
      if (shapeRows.some((r) => r !== null)) {
        let meanDiag = 0
        for (let a = 0; a < A.P; a++) meanDiag += fit.XtWX[a * A.P + a] / A.P
        const kappa = constraintWeight * meanDiag
        for (; shapeIterations < 50;) {
          const viol = violationsOf(fit.beta, 1e-10)
          const fresh = viol.filter(([j, r]) => !active[j].has(r))
          if (fresh.length === 0) break
          fresh.forEach(([j, r]) => active[j].add(r))
          const extra = new Float64Array(A.P * A.P)
          A.terms.forEach((t, j) => {
            const R = shapeRows[j]
            if (!R) return
            for (const r of active[j])
              for (let a = 0; a < t.size; a++)
                for (let b = 0; b < t.size; b++)
                  extra[(A.offsets[j] + a) * A.P + A.offsets[j] + b] += kappa * R[r * t.size + a] * R[r * t.size + b]
          })
          fit = fitAt(A, data, family, lk, penaltyMatrix(A, lambdas, extra), maxSteps, fit.beta)
          shapeIterations++
        }
      }

      const { final, beta, Hinv } = fit
      const n = A.n
      const mu = f64(final.mu)
      const y = f64(data.y)
      const w = data.weights ? f64(data.weights) : new Float64Array(n).fill(1)
      const V = f64(family.variance(final.mu) as Tensor)
      let pearson = 0
      for (let i = 0; i < n; i++) pearson += (w[i] * (y[i] - mu[i]) ** 2) / V[i]
      const dispersion = family.dispersion ?? pearson / (n - fit.edf)
      const cov = Float64Array.from(Hinv, (v) => v * dispersion)
      const termEdf = A.terms.map((t, j) => {
        let s = 0
        for (let c = 0; c < t.size; c++) s += fit.edfDiag[A.offsets[j] + c]
        return s
      })

      // Rows of the model matrix at new inputs.
      const designAt = (x: Tensor): F64 => {
        const [m, d] = x.shape
        if (d !== A.d) throw new Error(`gam: fitted on ${A.d} features, given ${d}`)
        const Xin = f64(x)
        const out = new Float64Array(m * A.P)
        for (let i = 0; i < m; i++) out[i * A.P] = 1
        A.terms.forEach((t, j) => {
          const B = times(t.raw(Xin, m, d), m, t.rawSize, t.Z, t.size)
          for (let i = 0; i < m; i++)
            for (let c = 0; c < t.size; c++) out[i * A.P + A.offsets[j] + c] = B[i * t.size + c]
        })
        return out
      }
      const forward = (x: Tensor) => {
        const Xd = designAt(x)
        const m = x.shape[0]
        const eta = new Float64Array(m)
        for (let i = 0; i < m; i++) for (let a = 0; a < A.P; a++) eta[i] += Xd[i * A.P + a] * beta[a]
        return vec(eta)
      }
      // A term's block of the design on a grid of its own feature(s).
      const termGrid = (j: number, grid: Tensor): { B: F64; m: number } => {
        const t = A.terms[j]
        const g = f64(grid)
        const m = grid.shape[0]
        const rows = new Float64Array(m * A.d)
        const width = grid.shape.length === 2 ? grid.shape[1] : 1
        for (let i = 0; i < m; i++) {
          for (let k = 0; k < Math.min(width, t.spec.kind === 'tensor' ? 2 : 1); k++)
            rows[i * A.d + t.features[k]] = g[i * width + k]
          if (t.spec.kind === 'smooth' && t.spec.by !== undefined)
            rows[i * A.d + t.spec.by] = t.byLevel === null || t.byLevel === undefined ? 1 : t.byLevel
        }
        return { B: times(t.raw(rows, m, A.d), m, t.rawSize, t.Z, t.size), m }
      }
      const blockCov = (j: number) => {
        const t = A.terms[j]
        const o = A.offsets[j]
        const C = new Float64Array(t.size * t.size)
        for (let a = 0; a < t.size; a++) for (let b = 0; b < t.size; b++) C[a * t.size + b] = cov[(o + a) * A.P + o + b]
        return C
      }
      const partial = (j: number, grid: Tensor): PartialEffect => {
        const t = A.terms[j]
        if (!t) throw new Error(`gam: no term ${j}`)
        const { B, m } = termGrid(j, grid)
        const C = blockCov(j)
        const fitv = new Float64Array(m)
        const se = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let f = 0
          let v = 0
          for (let a = 0; a < t.size; a++) {
            f += B[i * t.size + a] * beta[A.offsets[j] + a]
            for (let b = 0; b < t.size; b++) v += B[i * t.size + a] * C[a * t.size + b] * B[i * t.size + b]
          }
          fitv[i] = f
          se[i] = Math.sqrt(Math.max(v, 0))
        }
        return { fit: vec(fitv), se: vec(se) }
      }
      const partialDraws = (s: Stream, j: number, grid: Tensor, count: number) => {
        const t = A.terms[j]
        const { B, m } = termGrid(j, grid)
        const { L } = cholesky(fromData(blockCov(j), [t.size, t.size]))
        const Lf = f64(L)
        const z = f64(normals(s, [count, t.size]))
        const out = new Float64Array(count * m)
        for (let r = 0; r < count; r++) {
          const b = new Float64Array(t.size)
          for (let a = 0; a < t.size; a++) {
            let v = beta[A.offsets[j] + a]
            for (let c = 0; c <= a; c++) v += Lf[a * t.size + c] * z[r * t.size + c]
            b[a] = v
          }
          for (let i = 0; i < m; i++) for (let a = 0; a < t.size; a++) out[r * m + i] += B[i * t.size + a] * b[a]
        }
        return fromData(out, [count, m])
      }
      const partialResiduals = (j: number) => {
        const t = A.terms[j]
        if (!t) throw new Error(`gam: no term ${j}`)
        const dmu = f64(lk.derivative(final.eta) as Tensor)
        const out = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let f = 0
          for (let c = 0; c < t.size; c++) f += A.X[i * A.P + A.offsets[j] + c] * beta[A.offsets[j] + c]
          out[i] = f + (y[i] - mu[i]) / dmu[i]
        }
        return vec(out)
      }
      const meanAt = (x: Tensor) => lk.inverse(forward(x)) as Tensor
      const base = {
        kind: 'model' as const,
        name: 'gam' as const,
        family,
        link: lk,
        terms: A.terms,
        labels: A.terms.map((t) => t.label),
        coefficients: vec(beta),
        intercept: beta[0],
        lambdas,
        edf: fit.edf,
        termEdf,
        dispersion,
        deviance: final.deviance,
        smoothingScore: { method, value: score, evaluations },
        covariance: fromData(cov, [A.P, A.P]),
        fitted: final.mu,
        linearPredictor: final.eta,
        converged: final.converged,
        jitter: fit.jitter,
        shape: { iterations: shapeIterations, violations: violationsOf(beta, 1e-6).length },
        training: fit.training,
        partial,
        partialResiduals,
        residuals: (kind?: ResidualKind) =>
          residuals({ y: data.y, mu: final.mu, eta: final.eta, weights: data.weights, family, link: lk }, kind),
        partialDraws,
        forward,
        decide: meanAt,
        predictive: (x: Tensor) => family.predictive(meanAt(x), dispersion),
      }
      return withSampling(withExpectation(base)) as GamModel
    },
  }
}

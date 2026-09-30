/**
 * `aifn/gam`: generalised additive models on `aifn/smooth` and `aifn/glm`.
 *
 * ```ts
 * const m = gam({ terms: [s(0), s(1, { constraint: 'increasing' }), cyclic(2, { range: [0, 24] })] }).fit({ x, y })
 * m.partial(0, grid) // { fit, se }: the partial effect with pointwise standard errors
 * m.termEdf, m.lambdas, m.score
 * ```
 *
 * - Terms: `s` (P-spline), `cyclic`, `thinPlate` (Wood's regression spline), `te` (tensor product, one λ per
 *   direction), `linearTerm`, `factorTerm`; `by` variables (numeric: varying coefficients; `byFactor`: one smooth per
 *   level); shape constraints `increasing`, `decreasing`, `convex`, `concave` on P-splines..
 * - Fitting: `gam` (penalised IRLS for any `aifn/glm` family and link; λ by REML/LAML or GCV/UBRE, or fixed; EDF per
 *   term; Bayesian covariance; `partial` effects with standard errors and `partialDraws`), `backfitting` (with local
 *   scoring, as a traceable `Algorithm`).
 * - Variants: `expectileGam` (asymmetric least squares, `expectileLaws` as a traceable `Algorithm`) and
 *   `explainableBoostingMachine` (cyclic boosting of one-split trees on binned features, `ebmBoosting`).
 */

export {
  cyclic,
  factorTerm,
  linearTerm,
  s,
  te,
  thinPlate,
  type BuiltTerm,
  type ShapeConstraint,
  type SmoothOptions,
  type TermSpec,
} from './terms'
export { gam, type GamData, type GamModel, type GamParams, type PartialEffect, type SmoothingMethod } from './model'
export { backfitting, type BackfitProblem, type BackfitState } from './backfit'
export { expectileGam, expectileLaws, type ExpectileGamParams, type ExpectileState } from './expectile'
export { ebmBoosting, explainableBoostingMachine, type EbmModel, type EbmParams, type EbmState } from './ebm'

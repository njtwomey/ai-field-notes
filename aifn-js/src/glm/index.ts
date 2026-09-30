/**
 * `aifn/glm`: generalised linear models.
 *
 * ```ts
 * const m = glm({ family: poisson() }).fit({ x, y, offset })
 * m.coefficients, m.standardErrors, m.pValues, m.deviance, m.residuals('pearson')
 * m.training // the IRLS trace
 * ```
 *
 * - Links (`link(name)`): identity, log, logit, probit, cloglog, inverse, inverse-squared, sqrt; each has `link`,
 *   `inverse` and `derivative` built from primitives (numbers, tensors, traced values).
 * - Families: `gaussian`, `binomial` (Bernoulli, or proportions with trials as weights), `poisson`, `gamma`,
 *   `inverseGaussian`, `negativeBinomial(θ)`; `family(name)`. Each has `variance`, `unitDeviance`,
 *   `logLikelihood` and a `predictive` distribution.
 * - Fitting: `irls` (penalised IRLS as a traceable `Algorithm`, shared with `aifn/gam`), `weightedSolve`, `deviance`.
 * - Estimators: `glm` (weights, offsets, optional ridge; Wald standard errors, statistics and p-values, dispersion,
 *   deviance and null deviance, AIC, residuals), `negativeBinomialRegression` (θ by ML, `thetaMaximumLikelihood`),
 *   `multinomialLogisticRegression` (wraps `aifn/estimators`' `logisticRegression`, adds covariance and contrasts).
 */

export {
  binomial,
  family,
  gamma,
  gaussian,
  inverseGaussian,
  link,
  negativeBinomial,
  poisson,
  type Family,
  type FamilyName,
  type Link,
  type LinkName,
} from './families'
export { deviance, irls, weightedSolve, type IrlsProblem, type IrlsState } from './irls'
export {
  glm,
  negativeBinomialRegression,
  thetaMaximumLikelihood,
  type GlmData,
  type GlmModel,
  type GlmParams,
  type NegativeBinomialParams,
  type ResidualKind,
} from './model'
export {
  multinomialLogisticRegression,
  type Contrasts,
  type MultinomialModel,
  type MultinomialParams,
} from './multinomial'

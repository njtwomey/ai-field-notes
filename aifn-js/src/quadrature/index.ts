/**
 * `aifn/quadrature`: numerical integration with error estimates.
 *
 * - Fixed rules: `trapezoid`, `simpson` (composite, equal panels), `trapezoidSamples` (numpy's `trapezoid`),
 *   `integrateGauss`.
 * - Gaussian rules (nodes and weights, ascending): `gaussLegendre`, `gaussHermite` (physicists' or probabilists'),
 *   `gaussLaguerre` (generalised); `normalExpectation` for E f(X), X ~ N(mean, sd²).
 * - Traceable adaptive methods: `romberg`, `adaptiveSimpson`, `gaussKronrod` (7–15, globally adaptive, as QUADPACK's
 *   QAG); `kronrod15` for one interval; `integrate` (like scipy's `quad`, infinite limits allowed) and
 *   `integrateRomberg`.
 * - Several dimensions: `productRule`, `integrate2d`; `monteCarlo` (traceable, with standard errors) and
 *   `integrateMonteCarlo`; `halton`, `sobol` sequences and randomised `quasiMonteCarlo`.
 */

export {
  integrateRomberg,
  romberg,
  simpson,
  trapezoid,
  trapezoidSamples,
  type Integrand,
  type RombergState,
} from './rules'
export {
  gaussHermite,
  gaussLaguerre,
  gaussLegendre,
  integrateGauss,
  normalExpectation,
  type QuadratureRule,
} from './gauss'
export {
  adaptiveSimpson,
  gaussKronrod,
  integrate,
  kronrod15,
  type AdaptiveSimpsonState,
  type GaussKronrodState,
  type IntegrationResult,
  type Interval,
} from './adaptive'
export {
  halton,
  integrate2d,
  integrateMonteCarlo,
  monteCarlo,
  productRule,
  quasiMonteCarlo,
  sobol,
  type MonteCarloOptions,
  type MonteCarloResult,
  type MonteCarloState,
  type MultivariateIntegrand,
} from './multivariate'

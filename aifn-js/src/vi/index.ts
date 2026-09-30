/**
 * `aifn/vi`: variational inference. Every method minimises the reverse divergence KL(q ‖ p), equivalently maximises
 * the evidence lower bound ELBO = E_q[log p̃] + H[q] = log Z − KL(q ‖ p).
 *
 * - Families: `meanFieldGaussian(d)`, `fullRankGaussian(d)` with flat parameters λ, reparameterisation, entropy and
 *   score kernels.
 * - The ELBO and its gradient estimators: `elbo` (with standard error), `elboGradient` (reparameterisation or score
 *   function, with `none`, `leave-one-out` or `control-variate` baselines), `gradientVariance`.
 * - Black-box VI: `bbvi(target, { family, estimator, lr })`, an `Algorithm` driving `aifn/optim`'s Adam.
 * - Coordinate ascent: `caviNormalGamma` (Gaussian with unknown mean and precision, with the exact posterior
 *   `normalGammaPosterior` for comparison) and `caviGaussianMixture` (Bishop §10.2) with `mixturePredictiveDensity`.
 */

export { fullRankGaussian, meanFieldGaussian, type FamilyKernels, type GaussianFamily, type VectorLike } from './family'
export {
  elbo,
  elboGradient,
  gradientVariance,
  type Baseline,
  type ElboEstimate,
  type ElboGradient,
  type ElboGradientOptions,
  type GradientEstimator,
  type GradientVariance,
} from './elbo'
export { bbvi, type BbviOptions, type BbviStart, type BbviState } from './bbvi'
export {
  caviNormalGamma,
  defaultNormalGammaPrior,
  normalGammaPosterior,
  type CaviNormalGammaState,
  type NormalGammaPosterior,
  type NormalGammaPrior,
} from './cavi'
export { caviGaussianMixture, mixturePredictiveDensity, type MixturePrior, type MixtureState } from './mixture'

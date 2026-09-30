/**
 * `aifn/ep`: expectation propagation and assumed density filtering as a first-class module, usable with or without
 * `aifn/pgm` (plan §3).
 *
 * - Message algebra: univariate Gaussians in natural parameters (`naturalGaussian`, `gaussianMoments`,
 *   `multiplyGaussians`, `divideGaussians`, `powerGaussian`, `dampGaussian`, `gaussianToNormal`,
 *   `normalToGaussian`), multivariate ones (`naturalMvGaussian`, `mvGaussianMoments`, `multiplyMvGaussians`,
 *   `divideMvGaussians`), and exponential-family messages (`messageOf`, `multiplyMessages`, `divideMessages`,
 *   `dampMessages`, `powerMessage`, `messageToDistribution`).
 * - Tilted distributions (moment matching): `stepTilted`, `probitTilted`, `intervalTilted` (truncated Gaussian, the
 *   draw factor), `clutterTilted`, `tiltedByQuadrature` (any factor, any power).
 * - Algorithms: `expectationPropagationSteps` (EP and power EP, with damping and convergence diagnostics),
 *   `epLogEvidence`, `adfSteps`.
 * - Examples: the clutter problem (`clutterEp`, `clutterPosterior`, `clutterLogLikelihood`, `sampleClutter`),
 *   TrueSkill (`trueSkillUpdate`, `drawMargin`, `TRUESKILL_DEFAULTS`, `trueSkillEpSteps` over a match set), and the
 *   Bayes point machine (`bayesPointMachineSteps`, `bayesPointMachinePredict`).
 */

export {
  dampGaussian,
  dampMessages,
  divideGaussians,
  divideMessages,
  divideMvGaussians,
  gaussianMoments,
  gaussianToNormal,
  messageOf,
  messageToDistribution,
  multiplyGaussians,
  multiplyMessages,
  multiplyMvGaussians,
  mvGaussianMoments,
  naturalGaussian,
  naturalMvGaussian,
  normalToGaussian,
  powerGaussian,
  powerMessage,
  UNIFORM_GAUSSIAN,
  type ExpFamilyMessage,
  type GaussianMoments,
  type NaturalGaussian,
  type NaturalMvGaussian,
} from './gaussian'
export {
  clutterTilted,
  intervalTilted,
  probitTilted,
  stepTilted,
  tiltedByQuadrature,
  type ClutterFactorOptions,
  type ProbitOptions,
  type QuadratureTiltOptions,
  type Tilted,
} from './tilted'
export {
  adfSteps,
  epLogEvidence,
  expectationPropagationSteps,
  type AdfOptions,
  type AdfState,
  type EpOptions,
  type EpState,
  type TiltedFn,
} from './ep'
export {
  bayesPointMachinePredict,
  bayesPointMachineSteps,
  clutterEp,
  clutterLogLikelihood,
  clutterPosterior,
  drawMargin,
  sampleClutter,
  TRUESKILL_DEFAULTS,
  trueSkillEpSteps,
  trueSkillUpdate,
  type BayesPointMachineOptions,
  type BayesPointMachineState,
  type ClutterProblem,
  type Match,
  type Rating,
  type TrueSkillEpOptions,
  type TrueSkillEpState,
  type TrueSkillOptions,
  type TrueSkillUpdate,
} from './examples'

/**
 * `aifn/inference/expectation-propagation`: expectation propagation: Gaussian and exponential-family message algebra,
 * tilted moments (probit, step, interval, by quadrature), `ep`, assumed density filtering and the evidence.
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
  intervalTilted,
  lift,
  probitTilted,
  stepTilted,
  tiltedByQuadrature,
  type Out,
  type ProbitOptions,
  type QuadratureTiltOptions,
  type Tilted,
} from './tilted'
export {
  assumedDensityFiltering,
  epLogEvidence,
  expectationPropagation,
  type AdfOptions,
  type AdfState,
  type EpOptions,
  type EpState,
  type TiltedFn,
} from './ep'

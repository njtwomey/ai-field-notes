/**
 * `aifn/info`: information theory (plan §3).
 *
 * - Measures of discrete distributions (vectors or joint tables, normalised here; nats unless `base` is given):
 *   `entropy`, `jointEntropy`, `conditionalEntropy`, `crossEntropy`, `klDivergence`, `jensenShannonDivergence`,
 *   `jensenShannonDistance`, `fDivergence` with `fGenerators`, `totalVariation`, `hellingerDistance`,
 *   `mutualInformation`, `pointwiseMutualInformation`; `differentialEntropy` of a distribution object. All but
 *   `fDivergence` are differentiable.
 * - Continuous mutual information: `gaussianMutualInformation` (closed form), `ksgMutualInformation` (the
 *   Kraskov–Stögbauer–Grassberger estimator from samples).
 * - Blahut–Arimoto: `blahutArimotoCapacity` and `blahutArimotoRateDistortion` (traceable `Algorithm`s), with
 *   `channelCapacity`, `rateDistortion` and `rateDistortionCurve`.
 * - KL projections onto the normal family: `normalProjection(p, 'forward' | 'reverse')` fits N(μ, σ²) to a univariate
 *   p by L-BFGS with autodiff gradients (moment projection, mode covering; information projection, mode seeking),
 *   keeping the optimiser's path.
 * - Coding: `huffmanCode` (with its `Tree`), `huffmanSteps` (traceable merges), `huffmanTree`, `shannonFanoCode`,
 *   `shannonCode`, `kraftSum`, `arithmeticInterval`,
 *   `hammingDistance`, `hammingWeight`, `minimumDistance`, and the `hammingBound`, `singletonBound` and
 *   `plotkinBound` on code sizes.
 */

export {
  asValue,
  conditionalEntropy,
  crossEntropy,
  differentialEntropy,
  entropy,
  fDivergence,
  fGenerators,
  hellingerDistance,
  jensenShannonDistance,
  jensenShannonDivergence,
  jointEntropy,
  klDivergence,
  mutualInformation,
  pointwiseMutualInformation,
  totalVariation,
  type BaseOption,
  type FGenerator,
  type Probabilities,
} from './measures'
export {
  normalProjection,
  type KlDirection,
  type NormalFitStep,
  type NormalProjection,
  type NormalProjectionOptions,
} from './projection'
export { gaussianMutualInformation, ksgMutualInformation, type SampleRows } from './continuous'
export {
  blahutArimotoCapacity,
  blahutArimotoRateDistortion,
  channelCapacity,
  rateDistortion,
  rateDistortionCurve,
  type CapacityOptions,
  type CapacityState,
  type ChannelCapacity,
  type MatrixInput,
  type RateDistortionOptions,
  type RateDistortionPoint,
  type RateDistortionState,
} from './channel'
export {
  arithmeticInterval,
  hammingBound,
  hammingDistance,
  hammingWeight,
  huffmanCode,
  huffmanSteps,
  huffmanTree,
  kraftSum,
  minimumDistance,
  plotkinBound,
  shannonCode,
  shannonFanoCode,
  singletonBound,
  type ArithmeticInterval,
  type HuffmanEdgeData,
  type HuffmanNodeData,
  type HuffmanState,
  type HuffmanTree,
  type PrefixCode,
  type Word,
} from './coding'

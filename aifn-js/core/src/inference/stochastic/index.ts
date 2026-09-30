/**
 * `aifn/inference/stochastic`: Monte Carlo inference: Metropolis–Hastings (random-walk, independence), Gibbs over
 * conditionals, slice sampling, Hamiltonian Monte Carlo and NUTS, MALA and SGLD, sequential Monte Carlo and the
 * particle filter, multi-chain runs and diagnostics (R̂, ESS, MCSE; the autocorrelation function is
 * `aifn/probability/stats`'s), and Gibbs sampling on discrete factor graphs (`factorGraphGibbs`) and on model
 * descriptions (`modelGibbs`, by enumeration and conjugate updates).
 */

export {
  factorGraphGibbs,
  gibbsMarginals,
  modelGibbs,
  type ConditionalKind,
  type FactorGraphGibbsOptions,
  type FactorGraphGibbsState,
  type ModelGibbsOptions,
  type ModelGibbsState,
} from './factorGibbs'
export type { AcceptRejectState, ChainStart, ChainState, LogDensity, VectorLike } from './types'
export {
  independenceMetropolis,
  metropolisHastings,
  randomWalkMetropolis,
  type IndependentProposal,
  type MetropolisState,
  type Proposal,
  type RandomWalkOptions,
} from './metropolis'
export {
  bivariateGaussianConditionals,
  gaussianConditionals,
  gibbs,
  sliceSampler,
  type Conditional,
  type GibbsOptions,
  type GibbsState,
  type SliceOptions,
  type SliceState,
} from './gibbs'
export {
  hmc,
  leapfrog,
  nuts,
  type HmcOptions,
  type HmcState,
  type Leapfrog,
  type NutsOptions,
  type NutsState,
} from './hamiltonian'
export {
  mala,
  sgld,
  unadjustedLangevin,
  type LangevinOptions,
  type LangevinState,
  type MinibatchModel,
  type SgldOptions,
  type SgldState,
} from './langevin'
export {
  particleFilter,
  resample,
  resamplingSchemes,
  temperedSmc,
  type FilterStart,
  type ParticleFilterOptions,
  type ParticleFilterState,
  type ResamplingScheme,
  type StateSpaceModel,
  type TemperedModel,
  type TemperedSmcOptions,
  type TemperedSmcState,
} from './smc'
export { sampleChains, type ChainsResult, type SampleChainsOptions } from './chains'
export {
  effectiveSampleSize,
  integratedAutocorrelationTime,
  monteCarloStandardError,
  splitRhat,
  summarise,
  type ChainSummary,
  type Chains,
  type EssMethod,
  type RhatMethod,
} from './diagnostics'

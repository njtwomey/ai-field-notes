/**
 * `aifn/mcmc`: Markov chain Monte Carlo and sequential Monte Carlo samplers as traceable algorithms, with the standard
 * convergence diagnostics.
 *
 * - Targets: `{ dim, logDensity(θ), grad?(θ) }`; the gradient defaults to `aifn/autodiff`. Examples with analytic
 *   gradients: `banana`, `gaussianTarget`, `gaussianMixtureTarget`, `funnel`.
 * - Samplers (each an `Algorithm<{ x0 }, State>`; step t draws from `stream.child(t)`): `metropolisHastings`,
 *   `randomWalkMetropolis`, `independenceMetropolis`; `gibbs` over full conditionals (`gaussianConditionals`,
 *   `bivariateGaussianConditionals`); `sliceSampler`; `hmc` (with `leapfrog`) and `nuts`; `unadjustedLangevin`,
 *   `mala`; `sgld`. States expose proposals, acceptance, trajectories, energies and brackets.
 * - Sequential Monte Carlo: `resample` (multinomial, stratified, systematic, residual), `essFromLogWeights`,
 *   `particleFilter` (bootstrap, adaptive resampling, evidence), `temperedSmc`.
 * - Many chains: `sampleChains` runs chain k on `stream.child(k)` and stacks draws m×n×d.
 * - Diagnostics: `autocorrelation` (FFT), `integratedAutocorrelationTime`, `effectiveSampleSize` (bulk, tail, mean),
 *   `splitRhat` (rank-normalised, split, basic), `monteCarloStandardError`, `importanceEss`, `summarise`.
 */

export type { AcceptRejectState, ChainStart, ChainState, Target, VectorLike } from './types'
export { banana, funnel, gaussianMixtureTarget, gaussianTarget } from './targets'
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
  essFromLogWeights,
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
  autocorrelation,
  effectiveSampleSize,
  importanceEss,
  integratedAutocorrelationTime,
  monteCarloStandardError,
  splitRhat,
  summarise,
  type ChainSummary,
  type Chains,
  type EssMethod,
  type RhatMethod,
} from './diagnostics'

/**
 * `aifn/bandits`: multi-armed and contextual bandits.
 *
 * - Environments: `bernoulliBandit`, `gaussianBandit`, `linearBandit` (fixed or random arms). Each round draws every
 *   arm's reward from its own stream, so policies compared on one stream see the same rewards.
 * - Policies (pure: `init`, `choose`, `update`): `uniformPolicy`, `exploreThenCommit`, `epsilonGreedy` (constant or
 *   decaying), `ucb1`, `klUcb`, `thompsonBernoulli`, `thompsonGaussian`, `exp3`, `linUcb` (α = 0 is greedy ridge),
 *   `linearThompson`; helpers `klBernoulli`, `klUcbIndex`.
 * - Running: `banditRun` (a traceable algorithm, one round per step), `regretCurves` (mean, sd and 10–90% bands of
 *   cumulative pseudo-regret over replicates), `laiRobbinsBound`.
 */

export {
  bernoulliBandit,
  gaussianBandit,
  linearBandit,
  type BanditEnvironment,
  type LinearBanditOptions,
} from './environments'
export {
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  linearThompson,
  linUcb,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
  type ArmStatistics,
  type BanditPolicy,
  type Choice,
  type Exp3State,
  type RidgeState,
} from './policies'
export { banditRun, laiRobbinsBound, regretCurves, type BanditState, type RegretCurves } from './simulate'

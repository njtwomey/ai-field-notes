/**
 * `aifn-applied/decisions/bandits`: multi-armed and contextual bandits: the environment protocol, policies (ε-greedy,
 * UCB1, KL-UCB, Thompson sampling, EXP3, LinUCB, linear Thompson) and regret simulation.
 */

export { type BanditEnvironment } from './environment'
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

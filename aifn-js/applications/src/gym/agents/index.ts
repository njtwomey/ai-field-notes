/**
 * `aifn-applied/gym/agents`: agents on the gym protocol. `random.ts` (the baseline), `bandits.ts` (the bandit
 * policies), `tabular.ts` (TD control, n-step SARSA, Monte Carlo control, TD(0) prediction, REINFORCE) and
 * `planning.ts` (value and policy iteration as traceable algorithms on an MDP's tables, and as planning agents); child:
 * `control` (classic control).
 */

export { randomAgent, type RandomAgentState } from './random'
export {
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  laiRobbinsBound,
  linearThompson,
  linUcb,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
  type ArmStatistics,
  type Exp3State,
  type RidgeState,
} from './bandits'
export {
  expectedSarsaAgent,
  greedyPath,
  monteCarloControlAgent,
  nStepSarsaAgent,
  qLearningAgent,
  reinforceAgent,
  sarsaAgent,
  tdControlAgent,
  tdPredictionAgent,
  type MonteCarloAgentState,
  type NStepAgentState,
  type ReinforceState,
  type TabularAgentState,
  type TabularOptions,
  type TdAgentState,
  type TdMethod,
  type TdPredictionState,
} from './tabular'
export {
  evaluatePolicy,
  greedyPolicy,
  policyEvaluation,
  policyIteration,
  policyIterationAgent,
  valueIteration,
  valueIterationAgent,
  valuesFromQ,
  type PlannerState,
  type PolicyIterationState,
  type ValueState,
} from './planning'
export {
  lineariseDynamics,
  pendulumLqr,
  swingUpAgent,
  type Linearisation,
  type PendulumPlant,
  type SwingUpOptions,
  type SwingUpState,
  type TorqueAction,
} from './control'
export {
  crossEntropyAgent,
  linearPolicyAgent,
  lqrBangBangAgent,
  type CrossEntropyOptions,
  type CrossEntropyState,
  type GenerationSummary,
  type LqrBangBangOptions,
  type LqrBangBangState,
} from './control'

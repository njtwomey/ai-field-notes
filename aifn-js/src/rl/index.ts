/**
 * `aifn/rl`: tabular reinforcement learning.
 *
 * - MDPs and environments: `TabularMdp` (plain data), `tabularMdp` (from dense arrays), `gridworld` (Russell and
 *   Norvig's stochastic grid), `cliffWalking`, `maze` (from text rows), `frozenLake`; grid helpers `cellState`,
 *   `stateCell`, `GRID_ACTIONS`, `isActive`.
 * - Planning (traceable, value table and greedy policy per sweep): `valueIteration`, `policyIteration`,
 *   `policyEvaluation`; `evaluatePolicy` (exact), `qFromValues`, `valuesFromQ`, `greedyPolicy`, `greedyActions`,
 *   `policyMatrix`.
 * - Learning (traceable, one episode per step): `tdControl` (`sarsa`, `qLearning`, `expectedSarsa`), `nStepSarsa`,
 *   `monteCarloControl`, `tdPrediction`, `reinforce`; `sampleOutcome`, `greedyPath`.
 */

export {
  cellState,
  cliffWalking,
  FROZEN_LAKE_MAPS,
  frozenLake,
  GRID_ACTION_NAMES,
  GRID_ACTIONS,
  gridworld,
  isActive,
  maze,
  stateCell,
  tabularMdp,
  type CellKind,
  type GridworldOptions,
  type MazeOptions,
  type Outcome,
  type TabularMdp,
} from './mdp'
export {
  evaluatePolicy,
  greedyActions,
  greedyPolicy,
  policyEvaluation,
  policyIteration,
  policyMatrix,
  qFromValues,
  valueIteration,
  valuesFromQ,
  type PolicyInput,
  type PolicyIterationState,
  type ValueState,
} from './planning'
export {
  expectedSarsa,
  greedyPath,
  monteCarloControl,
  nStepSarsa,
  qLearning,
  reinforce,
  sampleOutcome,
  sarsa,
  tdControl,
  tdPrediction,
  type EpisodeOptions,
  type LearnerState,
  type TdMethod,
} from './learning'

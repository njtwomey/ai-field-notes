/**
 * `aifn-applied/decisions`: sequential decisions: the environment loop (`rollout`, `episodes`, `randomAgent`),
 * multi-armed and contextual bandits, and tabular reinforcement learning (planning and learning on MDPs).
 */

export {
  episodes,
  randomAgent,
  rollout,
  type EpisodeState,
  type RandomAgentState,
  type RolloutOptions,
  type RolloutState,
  type RolloutTransition,
} from './rollout'
export { banditRun, ucb1 } from './bandits'
export { mdpEnvironment, tabularMdp } from './reinforcement-learning'
export { valueIteration } from './reinforcement-learning/planning'
export { qLearning, qLearningAgent } from './reinforcement-learning/learning'

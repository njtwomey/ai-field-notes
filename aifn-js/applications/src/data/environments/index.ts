/**
 * `aifn-applied/data/environments`: environments: bandit arms (Bernoulli, Gaussian, linear) and grid worlds for
 * reinforcement learning (gridworld, cliff walking, mazes, FrozenLake).
 */

export { bernoulliBandit, gaussianBandit, linearBandit, type LinearBanditOptions } from './environments'
export {
  type GridworldOptions,
  gridworld,
  cliffWalking,
  type MazeOptions,
  maze,
  MAZES,
  mazeEnvironment,
  type MazeEnvironmentOptions,
  FROZEN_LAKE_MAPS,
  frozenLake,
} from './gridworlds'

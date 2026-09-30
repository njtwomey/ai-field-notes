/**
 * `aifn/optim/derivative-free`: derivative-free methods: Nelder–Mead, CMA-ES, simulated annealing, and Brent and
 * golden-section `minimizeScalar`.
 */

export {
  nelderMead,
  type NelderMeadOperation,
  type NelderMeadOptions,
  type NelderMeadState,
  type NelderMeadTrial,
} from './nelderMead'
export {
  cmaEs,
  simulatedAnnealing,
  type CmaEsOptions,
  type CmaEsState,
  type SimulatedAnnealingOptions,
  type SimulatedAnnealingState,
} from './stochastic'
export { minimizeScalar, type MinimizeScalarOptions, type MinimizeScalarResult } from './scalar'

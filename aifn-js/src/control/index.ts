/**
 * `aifn/control`: linear control. The Kalman filter lives in `aifn/timeseries`.
 *
 * - State space: `stateSpace`, `poles`, `stateFeedback`, `discretise` (ZOH, Euler, Tustin), `simulate` (a traceable
 *   algorithm; exact between samples for held inputs), `stepResponse`, `impulseResponse`, `initialResponse`, `respond`.
 * - Structure: `controllability`, `observability` (Kalman rank tests), `lyapunov`, `controllabilityGramian`,
 *   `observabilityGramian`, `polynomialFromRoots`, `ackermann` (single-input pole placement).
 * - Riccati and LQR: `kleinman` and `hamiltonianSign` (CARE), `riccatiRecursion` and `doubling` (DARE), each a
 *   traceable algorithm; `lqr` and `dlqr` run them.
 * - Transfer functions: `transferFunction`, `tfToStateSpace`, `stateSpaceToTf`, `freqResponse`, `polesZeros`,
 *   `frequencyGrid`, `bode`, `margins`, `series`, `feedback`.
 * - PID: `pid`, a loop around a plant with filtered derivative, saturation, anti-windup and delay.
 */

export {
  discretise,
  impulseResponse,
  initialResponse,
  poles,
  respond,
  simulate,
  stateFeedback,
  stateSpace,
  stepResponse,
  type DiscretisationMethod,
  type Input,
  type Poles,
  type Response,
  type SimulationOptions,
  type SimulationState,
  type StateSpace,
  type StateSpaceInput,
} from './system'
export {
  ackermann,
  controllability,
  controllabilityGramian,
  lyapunov,
  observability,
  observabilityGramian,
  polynomialFromRoots,
  type Gramian,
  type LyapunovSolution,
  type PolePlacement,
  type PoleSet,
  type RankTest,
} from './structure'
export {
  dlqr,
  doubling,
  hamiltonianSign,
  kleinman,
  lqr,
  riccatiRecursion,
  type DoublingState,
  type LqrResult,
  type RiccatiOptions,
  type RiccatiProblem,
  type RiccatiState,
  type SignState,
} from './riccati'
export {
  bode,
  feedback,
  freqResponse,
  frequencyGrid,
  margins,
  polesZeros,
  series,
  stateSpaceToTf,
  tfToStateSpace,
  transferFunction,
  type Bode,
  type Margins,
  type TransferFunction,
} from './transfer'
export { pid, type AntiWindup, type PidGains, type PidOptions, type PidState } from './pid'

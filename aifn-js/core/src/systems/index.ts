/**
 * `aifn/systems`: the one linear time-invariant system type, `LtiSystem` (tf, zpk, ss, sos; continuous or discrete),
 * shared by `aifn/signal`'s filters and `aifn/dynamics`' control.
 *
 * - Construction: `transferFunction`, `zerosPolesGain`, `stateSpace`, `secondOrderSections`.
 * - Conversion (exact): `toTransferFunction`, `toZerosPolesGain`, `toStateSpace`, `toSecondOrderSections`, `convert`.
 * - Analysis: `poles`, `systemZeros`, `stability`, `dimensions`; `controllability`, `observability` (Kalman rank tests),
 *   `controllabilityGramian`, `observabilityGramian`.
 * - Responses: `frequencyResponse` (a `Spectrum`), `responseAt`, `frequencyGrid`, `bode`, `margins`; `simulate` (a
 *   traceable algorithm; exact between samples for held inputs), `respond`, `stepResponse`, `impulseResponse`,
 *   `initialResponse`.
 * - New systems: `discretise` (zoh, Euler, Tustin), `stateFeedback`, `series`, `parallel`, `feedback`.
 * - Feedback design: `placePoles` (multi-input robust pole placement, Kautsky–Nichols–Van Dooren).
 * - Zeros, poles and responses are complex128 tensors (`ComplexLike` inputs, from `aifn/numerics/polynomial`).
 */

export type { LtiSystem, Representation } from 'aifn/foundation/contracts'
export type { ComplexLike } from 'aifn/numerics/polynomial'
export {
  convert,
  dimensions,
  poles,
  secondOrderSections,
  stability,
  stateSpace,
  toSecondOrderSections,
  toStateSpace,
  toTransferFunction,
  toZerosPolesGain,
  transferFunction,
  systemZeros,
  zerosPolesGain,
  type ChannelOptions,
  type LtiOf,
  type SecondOrderSectionsForm,
  type Stability,
  type StateSpaceForm,
  type StateSpaceInput,
  type SystemOptions,
  type TransferFunctionForm,
  type ZerosPolesGainForm,
} from './system'
export {
  controllability,
  controllabilityGramian,
  observability,
  observabilityGramian,
  type Gramian,
  type RankTest,
} from './structure'
export {
  bode,
  frequencyGrid,
  frequencyResponse,
  impulseResponse,
  initialResponse,
  margins,
  respond,
  responseAt,
  simulate,
  stepResponse,
  type Bode,
  type Input,
  type Margins,
  type RespondOptions,
  type SimulationOptions,
  type SimulationState,
  type StandardResponseOptions,
} from './responses'
export { discretise, feedback, parallel, series, stateFeedback, type DiscretisationMethod } from './transform'
export { placePoles, type PlacePolesOptions, type PolePlacementResult } from './placement'
export { systemsAlgorithms } from './registry'

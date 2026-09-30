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
 * - Complex values until `complex128`: `complex` (the [k, 2] (re, im) layout and scalar arithmetic).
 */

export type { LtiSystem, Representation } from 'aifn/foundation/contracts'
export * as complex from './complex'
export { type Complex, type ComplexLike } from './complex'
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

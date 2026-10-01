/**
 * `aifn/signal/filters`: digital filters as `LtiSystem`s: FIR design by windowing (`firwin`, with Kaiser estimates),
 * IIR design (`butter`, `cheby1`, `cheby2`, `iirfilter`), filtering as compositions over `linearFilter` (`lfilter`,
 * `sosfilt`, `lfilterZi`, `sosfiltZi`, `filtfilt`; differentiable in the coefficients), frequency response (`freqz`,
 * a complex128 `Spectrum`), group delay and phase unwrapping. `filterDesignRegistry` lists the design methods.
 */

export {
  butter,
  cheby1,
  cheby2,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  iirfilter,
  kaiserAttenuation,
  kaiserBeta,
  kaiserOrder,
  lfilter,
  lfilterZi,
  sosfilt,
  sosfiltZi,
  unwrapPhase,
  type FilterCoefficients,
  type FilterOptions,
  type FilterSpec,
  type Filtered,
  type FiltfiltOptions,
  type FirwinOptions,
  type GroupDelay,
  type IirOptions,
  type ResponseOptions,
} from './filters'
export { filterDesignRegistry, filtersFunctions } from './registry'

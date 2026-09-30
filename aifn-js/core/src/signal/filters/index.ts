/**
 * `aifn/signal/filters`: digital filters as `LtiSystem`s: FIR design by windowing (`firwin`, with Kaiser estimates),
 * IIR design (`butter`, `cheby1`, `cheby2`, `iirfilter`), filtering (`lfilter`, including second-order sections,
 * `lfilterZi`, `filtfilt`), frequency response (`freqz`, a `Spectrum`), group delay and phase unwrapping.
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
  unwrapPhase,
  type FirwinOptions,
  type GroupDelay,
  type IirOptions,
  type ResponseOptions,
} from './filters'

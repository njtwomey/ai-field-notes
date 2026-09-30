/**
 * `aifn/signal/decompositions`: empirical mode decomposition (`emd`, `eemd`) as `Decomposition`s, sifting and its
 * steps.
 */

export {
  eemd,
  emd,
  extrema,
  RILLING_RULE,
  siftImf,
  siftSteps,
  type EmdOptions,
  type SiftOptions,
  type SiftState,
  type StopRule,
} from './emd'

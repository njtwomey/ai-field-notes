/**
 * `aifn/text/cooccurrence`: windowed word–context counts, PMI, PPMI with context smoothing and shifting, and word
 * vectors from a truncated SVD.
 */

export {
  cooccurrence,
  pmi,
  ppmi,
  wordVectors,
  type Cooccurrence,
  type CooccurrenceOptions,
  type PmiOptions,
} from './cooccurrence'
export { cooccurrenceFunctions } from './registry'

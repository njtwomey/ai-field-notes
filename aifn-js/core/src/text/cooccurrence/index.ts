/**
 * `aifn/text/cooccurrence`: windowed word–context counts, PMI, PPMI with context smoothing and shifting, the
 * truncated SVD, and word vectors from it.
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
export { truncatedSvd, type TruncatedSvd } from './svd'
export { cooccurrenceFunctions } from './registry'

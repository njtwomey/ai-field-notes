/**
 * `aifn/text/features`: text as numbers. Word and character n-grams, the bag of words (counts or presence), TF-IDF
 * with SMART's term-frequency, document-frequency and normalisation variants, BM25 and BM25+, and signed feature
 * hashing with scikit-learn's MurmurHash3 conventions.
 */

export { characterNgrams, wordNgrams, type CharacterNgramOptions, type NgramRange } from './ngrams'
export { bagOfWords, type BagOfWords, type BagOfWordsOptions } from './bag'
export {
  bm25,
  bm25Weights,
  documentFrequency,
  inverseDocumentFrequency,
  smartWeighting,
  termFrequency,
  tfidf,
  type Bm25Options,
  type IdfScheme,
  type NormScheme,
  type TfidfOptions,
  type TfScheme,
} from './weighting'
export { featureHash, hashColumn, hashedFeatures, murmurHash3, type HashingOptions } from './hashing'
export { featuresFunctions } from './registry'

/** The functions of `aifn/text/features`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as bag from './bag'
import * as hashing from './hashing'
import * as ngrams from './ngrams'
import * as weighting from './weighting'

const fn = definer<FunctionInfo>('function', 'text/features')
const TFIDF = ['term-frequency-inverse-document-frequency', 'term-weighting-variants']

fn(
  {
    key: 'wordNgrams',
    name: 'Word n-grams',
    role: 'transform',
    summary: 'Runs of n consecutive tokens, for n in a range.',
    notes: ['word-n-grams'],
    cite: ['jurafsky2025'],
  },
  ngrams.wordNgrams,
)
fn(
  {
    key: 'characterNgrams',
    name: 'Character n-grams',
    role: 'transform',
    summary: 'Runs of n characters across the text, or inside space-padded words.',
    notes: ['character-n-grams-and-shingles'],
    cite: ['bojanowski2017'],
  },
  ngrams.characterNgrams,
)
fn(
  {
    key: 'bagOfWords',
    name: 'Bag of words',
    role: 'transform',
    returns: 'bag-of-words',
    summary: 'The document–term matrix of counts or presence over a vocabulary.',
    notes: ['bag-of-words'],
    cite: ['salton1975', 'manning2008'],
  },
  bag.bagOfWords,
)
fn(
  { key: 'documentFrequency', name: 'Document frequency', tex: '\\mathrm{df}_t', role: 'estimator', notes: TFIDF },
  weighting.documentFrequency,
)
fn(
  {
    key: 'inverseDocumentFrequency',
    name: 'Inverse document frequency',
    tex: '\\mathrm{idf}_t',
    role: 'estimator',
    summary: 'Standard, smoothed, probabilistic, Robertson–Spärck Jones and non-negative IDF.',
    notes: TFIDF,
    cite: ['sparckjones1972', 'robertson2009', 'pedregosa2011'],
  },
  weighting.inverseDocumentFrequency,
)
fn(
  {
    key: 'termFrequency',
    name: 'Term-frequency weighting',
    role: 'transform',
    summary: 'Raw, binary, logarithmic, augmented and log-average term frequency.',
    notes: ['term-weighting-variants'],
    cite: ['salton1988', 'manning2008'],
  },
  weighting.termFrequency,
)
fn(
  {
    key: 'tfidf',
    name: 'TF-IDF',
    role: 'transform',
    summary: 'Term frequency × inverse document frequency, each document normalised.',
    notes: TFIDF,
    cite: ['salton1988', 'sparckjones1972', 'pedregosa2011'],
  },
  weighting.tfidf,
)
fn(
  {
    key: 'smartWeighting',
    name: 'SMART weighting code',
    role: 'construction',
    summary: 'TF-IDF options from a SMART code such as ltc.',
    notes: ['term-weighting-variants'],
    cite: ['salton1988', 'manning2008'],
  },
  weighting.smartWeighting,
)
fn(
  {
    key: 'bm25Weights',
    name: 'BM25 term weights',
    role: 'transform',
    summary: 'IDF times saturated, length-normalised term frequency, per document and term; BM25+ with δ.',
    notes: ['term-frequency-inverse-document-frequency'],
    cite: ['robertson2009'],
  },
  weighting.bm25Weights,
)
fn(
  {
    key: 'bm25',
    name: 'BM25 scores',
    role: 'transform',
    summary: "Each document's BM25 (or BM25+) score for a query.",
    notes: ['term-frequency-inverse-document-frequency'],
    cite: ['robertson2009'],
  },
  weighting.bm25,
)
fn(
  {
    key: 'murmurHash3',
    name: 'MurmurHash3 (x86, 32-bit)',
    role: 'transform',
    summary: 'The 32-bit MurmurHash3 of a string’s UTF-8 bytes, as scikit-learn hashes features.',
    notes: ['feature-hashing-for-text'],
    cite: ['pedregosa2011'],
  },
  hashing.murmurHash3,
)
fn(
  { key: 'hashColumn', name: 'Hashed column and sign', role: 'transform', notes: ['feature-hashing-for-text'] },
  hashing.hashColumn,
)
fn(
  {
    key: 'hashedFeatures',
    name: 'Hashed features of a document (sparse)',
    role: 'transform',
    summary: 'The non-zero columns and signed counts of one hashed document.',
    notes: ['feature-hashing-for-text'],
    cite: ['weinberger2009'],
  },
  hashing.hashedFeatures,
)
fn(
  {
    key: 'featureHash',
    name: 'Feature hashing',
    role: 'transform',
    summary: 'The signed hashed document–feature matrix, as scikit-learn’s HashingVectorizer.',
    notes: ['feature-hashing-for-text'],
    cite: ['weinberger2009', 'pedregosa2011'],
  },
  hashing.featureHash,
)

/** The functions of the module, keyed by name. */
export const featuresFunctions = entries<FunctionInfo>('function', ngrams, bag, weighting, hashing) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>

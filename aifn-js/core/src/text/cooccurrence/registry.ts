/** The functions of `aifn/text/cooccurrence`, registered with the notes that define them. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as c from './cooccurrence'

const fn = definer<FunctionInfo>('function', 'text/cooccurrence')
const NOTE = 'co-occurrence-matrices-and-pointwise-mutual-information'

fn(
  {
    key: 'cooccurrence',
    name: 'Co-occurrence matrix',
    role: 'estimator',
    returns: 'cooccurrence',
    summary: 'Word × context counts within a window, uniform, harmonic or linear in distance.',
    notes: [NOTE, 'distributional-semantics'],
    cite: ['turney2010'],
  },
  c.cooccurrence,
)
fn(
  {
    key: 'pmi',
    name: 'Pointwise mutual information',
    tex: '\\operatorname{PMI}(w, c)',
    role: 'transform',
    summary: 'log p(w, c) / (p(w) P_α(c)) for every cell of a count matrix.',
    notes: [NOTE],
    glossary: 'pmi',
    cite: ['church1990', 'levy2015'],
  },
  c.pmi,
)
fn(
  {
    key: 'ppmi',
    name: 'Positive (shifted) PMI',
    tex: '\\max(\\operatorname{PMI} - \\log k, 0)',
    role: 'transform',
    summary: 'PMI clipped at zero, optionally shifted by log k (SPPMI).',
    notes: [NOTE, 'word-embeddings'],
    cite: ['levy2015', 'levy2014'],
  },
  c.ppmi,
)
fn(
  {
    key: 'wordVectors',
    name: 'Word vectors by truncated SVD',
    role: 'transform',
    summary: 'The rows of U_d Σ_d^p of a word × context matrix.',
    notes: [NOTE, 'word-embeddings', 'latent-semantic-analysis'],
    cite: ['levy2015'],
  },
  c.wordVectors,
)

/** The functions of the module, keyed by name. */
export const cooccurrenceFunctions = entries<FunctionInfo>('function', c) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>

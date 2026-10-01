/**
 * The registry of `aifn-applied/inference/topic-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as lda from './lda'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/topic-models')
const fn = definer<FunctionInfo>('function', 'inference/topic-models')
const notes = ['latent-dirichlet-allocation']

algorithm(
  {
    key: 'ldaCollapsedGibbs',
    name: 'LDA by collapsed Gibbs sampling',
    summary: 'Resamples each token’s topic from its full conditional with θ and φ integrated out; one sweep per step.',
    problem: 'factor-graph',
    state: { iterate: 'docTopic', objective: 'logLikelihood', flags: [] },
    random: true,
    notes: [...notes, 'gibbs-sampling'],
    cite: ['griffiths2004', 'blei2003'],
  },
  lda.ldaCollapsedGibbs,
)
fn({ key: 'ldaModel', name: 'LDA model specification', role: 'construction', notes, cite: ['blei2003'] }, lda.ldaModel)
fn({ key: 'matchLda', name: 'Match a model to LDA', role: 'property', notes }, lda.matchLda)
fn({ key: 'ldaOptions', name: 'LDA options from a model', role: 'construction', notes }, lda.ldaOptions)
fn(
  {
    key: 'ldaEstimates',
    name: 'LDA point estimates',
    summary: 'Posterior-mean topic–word and document–topic matrices from the counts of a Gibbs state.',
    role: 'estimator',
    notes,
    cite: ['griffiths2004'],
  },
  lda.ldaEstimates,
)

/** The algorithms of the module. */
export const topicModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  lda,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const topicModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', lda) as Table<FunctionInfo>

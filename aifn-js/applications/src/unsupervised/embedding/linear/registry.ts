/** The registry of `aifn-applied/unsupervised/embedding/linear`. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as andrews from './andrews'
import * as linear from './linear'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/linear')

definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/linear')(
  {
    key: 'smacofSteps',
    name: 'SMACOF',
    summary: 'Majorisation steps (the Guttman transform) that never increase metric stress.',
    problem: 'objective',
    state: { iterate: 'embedding', objective: 'stress', flags: ['converged'] },
    random: true,
    notes: ['multidimensional-scaling'],
    cite: ['kruskal1964', 'deleeuw2009'],
  },
  linear.smacofSteps,
)
fn(
  {
    key: 'classicalMds',
    name: 'Classical MDS',
    summary: 'The top eigenvectors of the double-centred squared distances.',
    role: 'fit',
    notes: ['multidimensional-scaling'],
    cite: ['torgerson1952'],
  },
  linear.classicalMds,
)
fn(
  {
    key: 'stress',
    name: 'Stress',
    role: 'estimator',
    notes: ['multidimensional-scaling', 'evaluating-embeddings'],
    cite: ['kruskal1964'],
  },
  linear.stress,
)
fn({ key: 'andrewsCurves', name: 'Andrews curves', role: 'transform', cite: ['andrews2003'] }, andrews.andrewsCurves)

/** The algorithms of the module, keyed by factory name. */
export const linearEmbeddingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  linear,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const linearEmbeddingFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  linear,
  andrews,
) as Table<FunctionInfo>

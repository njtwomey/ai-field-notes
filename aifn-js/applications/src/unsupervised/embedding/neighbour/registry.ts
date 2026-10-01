/** The registry of `aifn-applied/unsupervised/embedding/neighbour`: t-SNE and UMAP as traceable algorithms. */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as nnDescent from './nn-descent'
import * as tsne from './tsne'
import * as umap from './umap'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'unsupervised/embedding/neighbour')
const fn = definer<FunctionInfo>('function', 'unsupervised/embedding/neighbour')
const TSNE = ['t-distributed-stochastic-neighbour-embedding']
const UMAP = ['uniform-manifold-approximation-and-projection']

algorithm(
  {
    key: 'tsneSteps',
    name: 't-SNE',
    summary: 'Gradient steps with gains and early exaggeration on the KL divergence between neighbour distributions.',
    problem: 'objective',
    state: { iterate: 'embedding', objective: 'kl', flags: [] },
    random: true,
    notes: TSNE,
    cite: ['vandermaaten2008'],
  },
  tsne.tsneSteps,
)
algorithm(
  {
    key: 'umapSteps',
    name: 'UMAP layout',
    summary: 'One epoch of attractive edge samples and negative samples per step.',
    problem: 'graph',
    state: { iterate: 'embedding', flags: [] },
    random: true,
    notes: UMAP,
    cite: ['mcinnes2018'],
  },
  umap.umapSteps,
)
fn(
  { key: 'jointProbabilities', name: 't-SNE joint probabilities', role: 'construction', notes: TSNE },
  tsne.jointProbabilities,
)
fn(
  {
    key: 'perplexityCalibration',
    name: 'Perplexity calibration',
    summary: 'Per-point Gaussian bandwidths by bisection to a target perplexity.',
    role: 'solver',
    notes: TSNE,
  },
  tsne.perplexityCalibration,
)
fn({ key: 'fuzzyGraph', name: 'UMAP fuzzy graph', role: 'construction', notes: UMAP }, umap.fuzzyGraph)
fn({ key: 'curveParameters', name: 'UMAP curve parameters (a, b)', role: 'fit', notes: UMAP }, umap.curveParameters)
fn(
  {
    key: 'spectralLayout',
    name: 'Spectral initial layout',
    role: 'construction',
    notes: [...UMAP, 'laplacian-eigenmaps'],
  },
  umap.spectralLayout,
)
fn(
  {
    key: 'nearestNeighbourDescent',
    name: 'NN-descent',
    summary: 'Approximate k-nearest-neighbour graph by local joins of neighbours of neighbours.',
    role: 'solver',
    random: true,
    notes: ['graph-based-approximate-nearest-neighbours', ...UMAP],
  },
  nnDescent.nearestNeighbourDescent,
)

/** The algorithms of the module, keyed by factory name. */
export const neighbourEmbeddingAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  tsne,
  umap,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const neighbourEmbeddingFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  tsne,
  umap,
  nnDescent,
) as Table<FunctionInfo>

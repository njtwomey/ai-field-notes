/**
 * `aifn/embed`: dimensionality reduction and embeddings. Iterative methods are traceable `Algorithm`s (`…Steps`).
 *
 * - Linear and kernel: `pca` (SVD; explained variance, whitening, inverse transform), `kernelPca` (kernels from
 *   `aifn/kernels`).
 * - Scaling: `classicalMds`, `stress`, `metricMds` by SMACOF (`smacofSteps`).
 * - Neighbour graphs: `neighbourGraph`, `isomap` (Dijkstra geodesics from `aifn/graph`), `laplacianEigenmaps`,
 *   `locallyLinearEmbedding`.
 * - Stochastic neighbour embeddings: `tsne` (exact; `tsneSteps`, `perplexityCalibration`, `jointProbabilities`), `umap`
 *   (`fuzzyGraph`, `curveParameters`, `umapSteps`).
 * - Curves: `andrewsCurves` (Andrews, 1972), each row as a finite Fourier series, for plotting many dimensions.
 */

export {
  classicalMds,
  kernelPca,
  metricMds,
  pca,
  smacofSteps,
  stress,
  type ClassicalMds,
  type KernelPcaModel,
  type MetricMdsModel,
  type PcaModel,
  type SmacofState,
} from './linear'
export {
  isomap,
  laplacianEigenmaps,
  locallyLinearEmbedding,
  neighbourGraph,
  type IsomapModel,
  type LleModel,
  type SpectralEmbeddingModel,
} from './manifold'
export {
  jointProbabilities,
  perplexityCalibration,
  tsne,
  tsneSteps,
  type PerplexityCalibration,
  type TsneModel,
  type TsneParams,
  type TsneState,
} from './tsne'
export { curveParameters, fuzzyGraph, umap, umapSteps, type FuzzyGraph, type UmapModel, type UmapState } from './umap'
export { andrewsCurves } from './andrews'

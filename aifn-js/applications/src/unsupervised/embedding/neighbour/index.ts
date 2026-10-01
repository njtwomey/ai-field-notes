/**
 * `aifn-applied/unsupervised/embedding/neighbour`: neighbour embeddings: t-SNE and UMAP, and approximate k-nearest
 * neighbours by nearest-neighbour descent.
 */

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
export {
  curveParameters,
  DENSE_SPECTRAL_UP_TO,
  DESCENT_ABOVE,
  fuzzyGraph,
  spectralLayout,
  umap,
  umapSteps,
  type FuzzyGraph,
  type FuzzyGraphOptions,
  type NeighbourSearch,
  type UmapModel,
  type UmapState,
} from './umap'
export { nearestNeighbourDescent, type NearestNeighbourDescentOptions, type NeighbourLists } from './nn-descent'
export { neighbourEmbeddingAlgorithms, neighbourEmbeddingFunctions } from './registry'

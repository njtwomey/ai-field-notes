/**
 * `aifn-applied/unsupervised/embedding/manifold`: manifold learning: Isomap, locally linear embedding and Laplacian
 * eigenmaps.
 */

export {
  isomap,
  laplacianEigenmaps,
  locallyLinearEmbedding,
  neighbourGraph,
  type IsomapModel,
  type LleModel,
  type SpectralEmbeddingModel,
} from './manifold'
export { manifoldFunctions } from './registry'

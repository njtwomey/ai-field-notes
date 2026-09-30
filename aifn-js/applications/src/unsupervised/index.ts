/**
 * `aifn-applied/unsupervised`: unsupervised learning: clustering (k-means family, mixtures by EM, hierarchical,
 * density-based, spectral) and embeddings (linear, manifold, neighbour).
 */

export { kmeans, gaussianMixture, dbscan } from './clustering'
export { pca } from './embedding/linear'
export { tsne, umap } from './embedding/neighbour'

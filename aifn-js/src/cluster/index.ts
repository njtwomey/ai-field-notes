/**
 * `aifn/cluster`: clustering estimators in the `aifn/estimators` capability style (`decide` = a label; `predictive` =
 * responsibilities where the model is probabilistic). Iterative fits are traceable `Algorithm`s (`…Steps`).
 *
 * - Centroids: `kmeans` (Lloyd, `kmeansSteps`; restarts), `kmeansPlusPlus` (pick probabilities exposed),
 *   `miniBatchKMeans` (`miniBatchKMeansSteps`), `kMedoids` (PAM, `kMedoidsSteps`).
 * - Mixtures: `gaussianMixture` by EM (`gaussianMixtureSteps`; full, diagonal or spherical covariances;
 *   responsibilities per step).
 * - Hierarchies: `agglomerative` (`agglomerativeSteps`; single, complete, average, Ward), `linkage` (SciPy's format),
 *   `mergeTree` (the merges as a `{ nodes, root }` tree with heights), `cutTree`, `dendrogram`.
 * - Density: `dbscan` (core, border, noise), `optics` (ordering, reachability, `clustersAt(ε)`), `meanShift`
 *   (`meanShiftSteps`).
 * - Graphs: `spectralClustering` with `affinityMatrix` (RBF or k-nearest-neighbour), connectivity via `aifn/graph`.
 */

export {
  kMedoids,
  kMedoidsSteps,
  kmeans,
  kmeansPlusPlus,
  kmeansSteps,
  miniBatchKMeans,
  miniBatchKMeansSteps,
  type KMeansInit,
  type KMeansModel,
  type KMeansPlusPlus,
  type KMeansState,
  type KMedoidsModel,
  type KMedoidsState,
  type MiniBatchKMeansState,
} from './kmeans'
export {
  gaussianMixture,
  gaussianMixtureSteps,
  type CovarianceType,
  type GaussianMixtureModel,
  type MixtureInit,
  type MixtureParameters,
  type MixtureState,
} from './mixture'
export {
  agglomerative,
  agglomerativeSteps,
  cutTree,
  dendrogram,
  linkage,
  mergeTree,
  type AgglomerationState,
  type AgglomerativeModel,
  type Dendrogram,
  type Linkage,
  type MergeData,
  type MergeTree,
} from './hierarchical'
export {
  BORDER,
  CORE,
  NOISE,
  dbscan,
  meanShift,
  meanShiftSteps,
  optics,
  type DbscanModel,
  type MeanShiftModel,
  type MeanShiftState,
  type OpticsModel,
} from './density'
export { affinityMatrix, spectralClustering, type Affinity, type SpectralClusteringModel } from './spectral'
export { canonical as canonicalLabels } from './util'

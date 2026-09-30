/**
 * `aifn-applied/unsupervised/embedding/linear`: linear embeddings: PCA, kernel PCA, classical and metric MDS, and
 * Andrews curves.
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
export { andrewsCurves } from './andrews'

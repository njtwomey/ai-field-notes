/**
 * `aifn/losses`: training losses, each a composition of `aifn/tensor` and `aifn/special` primitives (so differentiable
 * through `aifn/autodiff`) carrying its metadata (`info`: key, display name, family, input kind, the note that defines
 * it and what its minimiser estimates), listed in `lossRegistry`.
 *
 * - Classification: `binaryCrossEntropyWithLogits` (with a positive-class weight), `binaryCrossEntropy`,
 *   `softmaxCrossEntropy` (integer labels or probability rows, label smoothing), `focalLoss`, `softmaxFocalLoss`; the
 *   margin surrogates `hinge`, `squaredHinge`, `logisticLoss`, `exponentialLoss`, `modifiedHuber` and their φ(m) in
 *   `surrogates` (with the 0–1 loss); the multiclass hinges `crammerSingerHinge` and `westonWatkinsHinge`.
 * - Regression: `meanSquaredError`, `meanAbsoluteError`, `huber`, `logCosh`, `pinball`, `poissonNll`, `gaussianNll`
 *   (by standard deviation).
 * - Ranking (one list [n] or lists [B, n] with graded relevance): pointwise `pointwiseBce`, `pointwiseSquaredError`;
 *   pairwise `rankNet`, `pairwiseHinge`, `bpr`, `lambdaRank` with `lambdaWeights`; listwise `listwiseSoftmax`,
 *   `listNet`, `listMle`, `approxNdcg`.
 * - Large output spaces and representations: `sampledSoftmax` (with the logQ correction), `negativeSampling`,
 *   `noiseContrastiveEstimation`, `inBatchSoftmax`, `infoNce`, `triplet`, `contrastive`.
 * - Divergences: `klLoss` (distribution objects), `jensenShannonLoss`, `distillation`.
 *
 * Conventions: predictions come first and targets second (the reverse of `aifn/metrics`, whose inputs are not
 * differentiated). Targets, labels, grades and sampling probabilities are constants. Every loss takes
 * `{ reduction: 'mean' | 'sum' | 'none' }` (default `mean`, as PyTorch); ranking losses sum within a list and reduce
 * over lists. Nothing is clamped: a probability of 0 on the true class gives +∞.
 */

export {
  defineLoss,
  isLoss,
  oneHot,
  type Loss,
  type LossFamily,
  type LossFunction,
  type LossInfo,
  type LossInput,
  type Reduction,
  type ReductionOptions,
  type Target,
} from './core'
export {
  binaryCrossEntropy,
  binaryCrossEntropyWithLogits,
  crammerSingerHinge,
  exponentialLoss,
  focalLoss,
  hinge,
  logisticLoss,
  modifiedHuber,
  softmaxCrossEntropy,
  softmaxFocalLoss,
  squaredHinge,
  surrogates,
  westonWatkinsHinge,
  type BinaryCrossEntropyOptions,
  type FocalOptions,
  type MulticlassHingeOptions,
  type SoftmaxCrossEntropyOptions,
  type SurrogateName,
} from './classification'
export {
  gaussianNll,
  huber,
  logCosh,
  meanAbsoluteError,
  meanSquaredError,
  pinball,
  poissonNll,
  type GaussianNllOptions,
  type HuberOptions,
  type PinballOptions,
  type PoissonNllOptions,
} from './regression'
export {
  approxNdcg,
  bpr,
  lambdaRank,
  lambdaWeights,
  listMle,
  listNet,
  listwiseSoftmax,
  pairwiseHinge,
  pointwiseBce,
  pointwiseSquaredError,
  rankNet,
  type ApproxNdcgOptions,
  type Gain,
  type LambdaOptions,
  type PairwiseHingeOptions,
  type RankNetOptions,
} from './ranking'
export {
  contrastive,
  inBatchSoftmax,
  infoNce,
  negativeSampling,
  noiseContrastiveEstimation,
  sampledSoftmax,
  triplet,
  type InfoNceOptions,
  type MarginOptions,
  type NceOptions,
  type SampledSoftmaxOptions,
} from './retrieval'
export { distillation, jensenShannonLoss, klLoss, type DistillationOptions } from './divergence'
export { getLoss, listLosses, lossRegistry } from './registry'

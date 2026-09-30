/**
 * `aifn/nn`: neural networks on `aifn/autodiff` (plan §8.1). Every layer is a composition of `aifn/tensor` and
 * `aifn/special` primitives, plus the primitives defined here with their vjps (gather and scatter-add, 2-D convolution
 * and its two adjoints, average pooling and its adjoint), so any function of a network's parameters is differentiable
 * by `grad` with no hand-written backward pass.
 *
 * - Parameters are pytrees of tensors (`Params`; `treeLeaves`, `treeMap`, `treeZip`, `countParams`).
 * - Functional forms: `linear`, `conv1d`, `conv2d`, `maxPool1d`, `maxPool2d`, `avgPool1d`, `avgPool2d`,
 *   `layerNorm`, `rmsNorm`, `batchNorm`, `dropout` (from a stream), `scaledDotProductAttention` (with masks and
 *   `causalMask`), `multiHeadAttention`, `takeRows` (embedding lookup), `gather`, `convOutputSize`.
 * - Layers (`Layer`: `init(stream)` → params, `apply(params, x, ctx)`): `Linear`, `Embedding`, `Conv1d`, `Conv2d`,
 *   `MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `Flatten`, `LayerNorm`, `RmsNorm`, `BatchNorm`, `Dropout`,
 *   `ActivationLayer`, `Sequential`, `Mlp`, `Residual`, `MultiHeadAttention`, `TransformerBlock`.
 * - Recurrent cells (`Cell`: `step(params, x, state)`): `RnnCell`, `GruCell`, `LstmCell`, and `unroll` over time.
 * - Activations: `relu` (a primitive), `leakyRelu`, `elu`, `gelu` (exact or tanh), `silu`, `identity`, and the
 *   `activations` table (sigmoid, tanh and softplus come from `aifn/special` and `aifn/tensor`).
 * - Initialisers on streams: `xavierUniform`, `xavierNormal`, `heUniform`, `heNormal`, `lecunUniform`, `normalInit`,
 *   `zerosInit`.
 * - Optimisers over pytrees: `sgd` (momentum, Nesterov, weight decay), `adam` (and AdamW via `decoupled`).
 * - Training: `training(...)` is a traceable `Algorithm` (minibatches and dropout from keyed substreams; loss, gradient
 *   norms per leaf and in total, parameter norm in every state).
 * - Inspection: `activations` records every layer's output; `inspect` adds the gradients of a loss with respect to
 *   each activation and parameter.
 *
 * Conventions: batch axes lead; features are last for dense and attention layers ([..., T, d]) and channels are axis
 * 1 for convolutions ([N, C, H, W], as PyTorch). Dense weights are stored [in, out] (PyTorch stores [out, in]).
 */

export { treeLeaves, treeMap, treeZip, countParams, type Leaf, type LeafValue, type Params } from './tree'
export {
  avgPool1d,
  avgPool2d,
  conv1d,
  conv2d,
  convOutputSize,
  gather,
  maxPool1d,
  maxPool2d,
  takeRows,
  type ConvOptions,
  type Pair,
  type PoolOptions,
} from './ops'
export {
  activationFn,
  activations as activationFunctions,
  elu,
  gelu,
  identity,
  leakyRelu,
  relu,
  silu,
  type Activation,
  type ActivationName,
} from './activations'
export {
  heNormal,
  heUniform,
  lecunUniform,
  normalInit,
  xavierNormal,
  xavierUniform,
  zerosInit,
  type Fans,
  type HeOptions,
  type Initialiser,
} from './init'
export {
  ActivationLayer,
  AvgPool1d,
  AvgPool2d,
  BatchNorm,
  batchNorm,
  Conv1d,
  Conv2d,
  Dropout,
  dropout,
  Embedding,
  Flatten,
  LayerNorm,
  layerNorm,
  Linear,
  linear,
  MaxPool1d,
  MaxPool2d,
  Mlp,
  Residual,
  RmsNorm,
  rmsNorm,
  Sequential,
  type BatchNormOptions,
  type Context,
  type ConvLayerOptions,
  type ConvParams,
  type EmbeddingParams,
  type Layer,
  type LinearOptions,
  type LinearParams,
  type MlpOptions,
  type NormParams,
} from './layers'
export {
  causalMask,
  multiHeadAttention,
  MultiHeadAttention,
  scaledDotProductAttention,
  TransformerBlock,
  type AttentionOptions,
  type AttentionResult,
  type MultiHeadAttentionParams,
  type MultiHeadOptions,
  type TransformerBlockOptions,
  type TransformerBlockParams,
} from './attention'
export {
  GruCell,
  LstmCell,
  RnnCell,
  unroll,
  type Cell,
  type CellOptions,
  type CellParams,
  type RecurrentState,
  type Unrolled,
} from './recurrent'
export {
  adam,
  sgd,
  type AdamOptions,
  type LearningRate,
  type Optimizer,
  type SgdOptions,
  type Slots,
} from './optimizers'
export { training, type Batch, type TrainingOptions, type TrainingState } from './train'
export { activations, inspect, type Activations, type Inspection } from './inspect'

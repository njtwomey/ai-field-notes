/**
 * `aifn/nn/layers`: layers as parameter trees with a forward function (after torch.nn.modules): `Linear`, `Embedding`,
 * convolutions and pools, `LayerNorm`, `RmsNorm`, `BatchNorm`, `Dropout`, `Sequential`, `Mlp`, `Residual`, multi-head
 * attention (`scaledDotProductAttention`, `MultiHeadAttention`) and RNN, GRU and LSTM cells.
 */

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
  type AttentionOptions,
  type AttentionResult,
  type MultiHeadAttentionParams,
  type MultiHeadOptions,
} from './attention'
export {
  GruCell,
  LstmCell,
  RnnCell,
  unrollRecurrent,
  type Cell,
  type CellOptions,
  type CellParams,
  type RecurrentState,
  type Unrolled,
} from './recurrent'

// For writing layers: record an activation (`tap`) and name a sub-layer's context (`childContext`).
export { childContext, tap } from './layers'

/**
 * `aifn/nn/layers`: layers as parameter trees with a forward function (after torch.nn.modules): `Linear`, `Embedding`,
 * convolutions and pools, `LayerNorm`, `RmsNorm`, `BatchNorm`, `Dropout`, `Sequential`, `Mlp`, `Residual`, and RNN, GRU
 * and LSTM cells, and the ODE block (`OdeBlock`, a neural ODE layer). Attention is `aifn/nn/attention`.
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
  type BatchNormBuffers,
  type BatchNormLayerOptions,
  type BatchNormOptions,
  type Buffers,
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
export { OdeBlock, type OdeBlockLayer, type OdeBlockOptions } from './ode'

// For writing layers: record an activation (`tap`) and name a sub-layer's context (`childContext`).
export { childContext, tap } from './layers'

/**
 * `aifn/nn/training`: the training loop (`trainingLoop`, a traceable `Algorithm` over any pytree update rule of
 * `aifn/optim/first-order`) and activation inspection (`activations`, `inspect`).
 */

export { trainingLoop, type Batch, type TrainingOptions, type TrainingState } from './train'
export { activations, inspect, type Activations, type Inspection } from './inspect'
export { trainingAlgorithms } from './registry'

/**
 * `aifn/nn`: neural networks, after torch.nn: functional operations, initialisers, layers and the training loop. Losses
 * are `aifn/learning/losses` (decision D6) and optimisers the pytree update rules of `aifn/optim/first-order`.
 * Children: functional, init, layers, training.
 */

export { relu, gelu, conv2d } from './functional'
export { xavierUniform, heNormal } from './init'
export { Linear, Mlp, Sequential, MultiHeadAttention } from './layers'
export { trainingLoop } from './training'

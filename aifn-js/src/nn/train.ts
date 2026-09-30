/**
 * Training as a traceable algorithm: minibatch gradient descent on a loss of the parameters, with any `Optimizer`.
 * Each step draws its minibatch and dropout masks from substreams keyed by the step number, so a step is a pure
 * function of its state and `seek`, `extend` and replays agree exactly.
 */

import { valueAndGrad } from 'aifn/autodiff'
import { permutation, stream, type Stream } from 'aifn/random'
import { fromData, mul, norm, toFlat, unwrap, type Tensor, type Value } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import type { Context } from './layers'
import { takeRows } from './ops'
import { adam, type Optimizer, type Slots } from './optimizers'
import { treeLeaves, treeMap, type Params } from './tree'

/** A dataset: named tensors whose first axis indexes the examples (all the same length), e.g. `{ x, y }`. */
export type Batch = Record<string, Tensor>

/** Options of `training`. */
export type TrainingOptions<P extends Params, B extends Batch> = {
  /**
   * The loss of parameters on a minibatch, a number or rank-0 value. `ctx` carries `train: true` and the step's
   * stream, to pass to layers (for dropout).
   */
  loss: (params: P, batch: B, ctx: Context) => Value
  /** The training set. */
  data: B
  /** Examples per step (default: the whole set, i.e. full-batch gradient descent). */
  batchSize?: number
  /** Default `adam({ lr: 0.01 })`. */
  optimizer?: Optimizer
  /**
   * Rescale the gradient when its global norm exceeds this (Pascanu, Mikolov & Bengio, 2013). The unclipped norm is
   * reported. Default: no clipping.
   */
  clipNorm?: number
  /** Flag divergence when the loss exceeds this or is not finite. Default 1e12. */
  divergeAbove?: number
}

/** The state of `training` at step t. */
export type TrainingState<P extends Params> = {
  t: number
  /** The parameters after t updates. */
  params: P
  /** The optimiser's slots (moments, velocity). */
  slots: Slots
  /** The loss of `params` on this step's minibatch (the one the next update uses). */
  loss: number
  /** Its gradient with respect to the parameters. */
  grads: P
  /** The global gradient norm ‖∇‖₂ over every parameter (before clipping). */
  gradNorm: number
  /** The gradient norm of each parameter leaf, by path (e.g. `[0].weight`). */
  gradNorms: Record<string, number>
  /** The global parameter norm ‖θ‖₂. */
  paramNorm: number
  /** Indices of this step's minibatch in the dataset (the whole set for full-batch training). */
  batch: Tensor | null
  /** The epoch this step's minibatch belongs to. */
  epoch: number
  stream: Stream
  diverged: boolean
}

/** The data's number of examples, checking every field agrees. */
function examplesOf(data: Batch): number {
  let n = -1
  for (const [name, t] of Object.entries(data)) {
    const m = t.shape[0]
    if (n >= 0 && m !== n) throw new Error(`training: data field '${name}' has ${m} rows, expected ${n}`)
    n = m
  }
  if (n <= 0) throw new Error('training: the data has no examples')
  return n
}

/**
 * The minibatch of step t: epochs are shuffled by `stream.child('epoch', e)` and cut into consecutive batches, so
 * every example is used once per epoch (the last partial batch of an epoch is dropped when it would be short).
 */
function minibatch(data: Batch, n: number, size: number, s: Stream, t: number) {
  if (size >= n) return { batch: data, indices: null, epoch: t }
  const perEpoch = Math.floor(n / size)
  const epoch = Math.floor(t / perEpoch)
  const k = t % perEpoch
  const order = toFlat(permutation(s.child('epoch', epoch), n)).slice(k * size, (k + 1) * size)
  const batch = Object.fromEntries(Object.entries(data).map(([key, v]) => [key, unwrap(takeRows(v, order)) as Tensor]))
  return { batch, indices: fromData(Int32Array.from(order), [size]), epoch }
}

const leafNorm = (v: Tensor | number) => (typeof v === 'number' ? Math.abs(v) : norm(v))

/**
 * Minibatch training of parameters on `loss` (Robbins & Monro, 1951; for the optimisers see `sgd` and `adam`). `init`
 * takes `{ params }` (e.g. `model.init(stream)`) and a stream for the minibatches and dropout (default
 * `stream('training')`); step t evaluates the loss and gradient on minibatch t and applies one optimiser update.
 *
 * Record what a figure needs with the trace's recorders, e.g. `{ loss: (s) => s.loss, gradNorm: (s) => s.gradNorm }`.
 */
export function training<P extends Params, B extends Batch>(
  options: TrainingOptions<P, B>,
): Algorithm<{ params: P }, TrainingState<P>> {
  const { loss, data, optimizer = adam({ lr: 0.01 }), clipNorm, divergeAbove = 1e12 } = options
  const n = examplesOf(data)
  const size = Math.min(options.batchSize ?? n, n)
  const lossAndGrad = valueAndGrad((params: P, batch: B, ctx: Context) => loss(params, batch, ctx))

  const evaluate = (params: P, t: number, s: Stream) => {
    const { batch, indices, epoch } = minibatch(data, n, size, s, t)
    const ctx: Context = { train: true, stream: s.child('step', t) }
    const { value, grad } = lossAndGrad(params, batch as B, ctx)
    const grads = grad as P
    const gradNorms: Record<string, number> = {}
    let total = 0
    for (const { path, value: g } of treeLeaves(grads)) {
      const v = leafNorm(g)
      gradNorms[path] = v
      total += v * v
    }
    const lossValue = typeof value === 'number' ? value : toFlat(value)[0]
    const paramNorm = Math.sqrt(treeLeaves(params).reduce((a, { value: p }) => a + leafNorm(p) ** 2, 0))
    return {
      loss: lossValue,
      grads,
      gradNorm: Math.sqrt(total),
      gradNorms,
      paramNorm,
      batch: indices,
      epoch,
      diverged: !Number.isFinite(lossValue) || Math.abs(lossValue) > divergeAbove,
    }
  }

  return {
    name: `training-${optimizer.name}`,
    init: ({ params }, s) => {
      const str = s ?? stream('training')
      return { t: 0, params, slots: optimizer.init(params), stream: str, ...evaluate(params, 0, str) }
    },
    step: (state) => {
      let grads: Params = state.grads
      if (clipNorm !== undefined && state.gradNorm > clipNorm) {
        const scale = clipNorm / state.gradNorm
        grads = treeMap(grads, (g) => (typeof g === 'number' ? g * scale : mul(g, scale)))
      }
      const { params, slots } = optimizer.update(state.params, grads, state.slots, state.t)
      const t = state.t + 1
      return { t, params: params as P, slots, stream: state.stream, ...evaluate(params as P, t, state.stream) }
    },
  }
}

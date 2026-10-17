/**
 * The training run behind the PointNet figure: a shared per-point MLP, a max pool and a classifier, trained to tell
 * noisy 2-D outlines of circles, squares and triangles apart. Layers, loss, optimiser, the pool, the critical set and
 * the data are all engine calls.
 */
import { stream } from 'aifn-compute/foundation/random'
import { reshape, tensor, toFlat, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { Mlp } from 'aifn-compute/nn/layers'
import { softmax } from 'aifn-compute/numerics/special'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { criticalSet, padSets, setPool } from 'aifn-compute/nn/sets'
import { shapeClouds } from 'aifn-methods/data/synthetic'

/** Points per cloud, clouds for training and for testing, and the width of the pooled feature. */
export const POINTS = 32
export const TRAIN = 30
export const TEST = 30
export const WIDTH = 16
/** Optimiser steps between checkpoints. */
const EVERY = 10

const PHI = Mlp([2, WIDTH, WIDTH], { outputActivation: 'relu' })
const RHO = Mlp([WIDTH, WIDTH, 3])
export type Params = { phi: ReturnType<typeof PHI.init>; rho: ReturnType<typeof RHO.init> }

export type Setup = { rotate: boolean; steps: number; rate: number; seed: number }
export type Checkpoint = { step: number; loss: number; train: number; test: number; params: Params }

const data = (rotate: boolean) => {
  const all = shapeClouds(stream(`pointnet/data/${rotate ? 'rotated' : 'upright'}`), {
    count: TRAIN + TEST,
    points: POINTS,
    noise: 0.06,
    rotate,
  })
  return {
    clouds: all.clouds,
    labels: all.labels,
    train: padSets(all.clouds.slice(0, TRAIN)),
    test: padSets(all.clouds.slice(TRAIN)),
  }
}
export const DATA = { upright: data(false), rotated: data(true) }

/** Per-point features φ(x_i) of a batch of clouds, `[B, K, WIDTH]`. */
function features(p: Params, x: Tensor): Value {
  const [B, K] = x.shape
  return reshape(PHI.apply(p.phi, reshape(x, [B * K, 2])), [B, K, WIDTH])
}

/** Class logits of a padded batch: ρ(max_i φ(x_i)). */
function logits(p: Params, b: { x: Tensor; mask: Tensor }): Value {
  return RHO.apply(p.rho, setPool(features(p, b.x), b.mask, 'max'))
}

const accuracy = (p: Params, b: { x: Tensor; mask: Tensor }, labels: readonly number[]) => {
  const rows = toRows(logits(p, b) as Tensor) as number[][]
  return rows.filter((r, i) => r.indexOf(Math.max(...r)) === labels[i]).length / labels.length
}

/** Full-batch Adam on the training clouds, with train and test accuracy every few steps. */
export function* pointNetRun(setup: Setup): Generator<Checkpoint> {
  const d = setup.rotate ? DATA.rotated : DATA.upright
  const yTrain = tensor(d.labels.slice(0, TRAIN))
  const alg = trainingLoop({
    loss: (p: Params) => softmaxCrossEntropy(logits(p, d.train), yTrain),
    data: { y: yTrain },
    optimizer: adamRule({ stepSize: setup.rate }),
  })
  const s = stream(`pointnet/init/${setup.seed}`)
  const it = live(alg, { params: { phi: PHI.init(s), rho: RHO.init(s) } }, { stream: s })
  let state: { params: Params; loss: number } | null = null
  for (let step = 0; step <= setup.steps; step++) {
    const next = it.next()
    if (!next.done && !next.value.state.diverged)
      state = { params: next.value.state.params, loss: next.value.state.loss }
    if (state && (step % EVERY === 0 || step === setup.steps))
      yield {
        step,
        loss: state.loss,
        train: accuracy(state.params, d.train, d.labels.slice(0, TRAIN)),
        test: accuracy(state.params, d.test, d.labels.slice(TRAIN)),
        params: state.params,
      }
  }
}

/** A fresh cloud of `points` points for the figure, from its own stream, with its label (cycling through the shapes). */
export function exampleCloud(rotate: boolean, k: number, points: number) {
  const all = shapeClouds(stream(`pointnet/example/${rotate ? 'rotated' : 'upright'}/${k}`), {
    count: 3,
    points,
    noise: 0.06,
    rotate,
  })
  return { cloud: all.clouds[k % 3], label: all.labels[k % 3] }
}

/**
 * One cloud through the trained network: its critical set (indices of the points that win at least one pooled
 * feature), the pooled feature of any subset of its points, and the class probabilities of a pooled feature.
 */
export function cloudModel(p: Params, cloud: readonly [number, number][]) {
  const full = toRows(
    reshape(features(p, tensor([cloud.map(([x, y]) => [x, y])])), [cloud.length, WIDTH]) as Tensor,
  ) as number[][]
  const pooled = (idx: readonly number[]) =>
    Array.from({ length: WIDTH }, (_, c) => (idx.length > 0 ? Math.max(...idx.map((i) => full[i][c])) : 0))
  const probabilities = (g: readonly number[]) =>
    (toRows(softmax(RHO.apply(p.rho, tensor([[...g]]))) as Tensor) as number[][])[0]
  return { critical: toFlat(criticalSet(tensor(full))), pooled, probabilities }
}

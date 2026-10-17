/**
 * The graph-classification run behind the readout figure: the same two-layer message passing network trained four
 * times, once per readout, on cycles, stars and random trees with constant node features. The layers and the training
 * are engine calls; the readouts come from this sub-chapter's pooling candidate.
 */
import { child, normals, stream } from 'aifn-compute/foundation/random'
import {
  add,
  matmul,
  mul,
  ones,
  tanh,
  tensor,
  toRows,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { propagate } from 'aifn-compute/graph/propagation'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { relu } from 'aifn-compute/nn/functional'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { attentionPool, batchGraphs, globalPool, type GraphBatch } from 'aifn-compute/nn/graph'
import { graphFamilies } from 'aifn-methods/data/synthetic'

export const READOUTS = ['sum', 'mean', 'max', 'attention'] as const
export type Readout = (typeof READOUTS)[number]
export type Task = 'family' | 'size'

/** Graphs in the training and test sets. */
export const TRAIN = 60
export const TEST = 30
/** Hidden width of both layers. */
const HIDDEN = 16
/** The size task's threshold: graphs with at least this many nodes are "large". */
export const LARGE = 10
/** Optimiser steps between checkpoints. */
const EVERY = 10

export type Setup = { task: Task; steps: number; rate: number; seed: number }
export type Checkpoint = {
  step: number
  loss: Record<Readout, number>
  train: Record<Readout, number>
  test: Record<Readout, number>
}

type Params = {
  w0: Tensor
  b0: Tensor
  w1: Tensor
  b1: Tensor
  gate: Tensor
  w2: Tensor
  b2: Tensor
  w3: Tensor
  b3: Tensor
}

/** The data: 90 graphs, the first 60 for training; labels for both tasks. */
export const DATA = graphFamilies(stream('graph-readout/data'), { count: TRAIN + TEST, minNodes: 6, maxNodes: 13 })
const labelsFor = (task: Task) => (task === 'family' ? [...DATA.family] : DATA.size.map((n) => (n >= LARGE ? 1 : 0)))
const TRAIN_BATCH = batchGraphs(DATA.graphs.slice(0, TRAIN))
const TEST_BATCH = batchGraphs(DATA.graphs.slice(TRAIN))

/**
 * Two GIN-0 layers $\hvec' = \tanh((\hvec + \sum_{u \in \Ncal(v)} \hvec_u)\Wmat + \bvec)$ from a constant
 * feature 1 on every node, the readout, and a one-hidden-layer ReLU classifier on the graph vector.
 */
function logits(p: Params, b: GraphBatch, readout: Readout): Value {
  let h: Value = ones([b.graph.nodes, 1])
  for (const [w, c] of [
    [p.w0, p.b0],
    [p.w1, p.b1],
  ] as const)
    h = tanh(add(matmul(add(h, propagate(b.graph, h)), w), c))
  const pooled =
    readout === 'attention'
      ? attentionPool(h, matmul(h, p.gate), b.batch, b.graphs).pooled
      : globalPool(h, b.batch, b.graphs, readout)
  return add(matmul(relu(add(matmul(pooled, p.w2), p.b2)), p.w3), p.b3)
}

function init(classes: number, seed: number): Params {
  const s = stream(`graph-readout/init/${seed}`)
  return {
    w0: normals(child(s, 'w0'), [1, HIDDEN]),
    b0: zeros([HIDDEN]),
    w1: mul(normals(child(s, 'w1'), [HIDDEN, HIDDEN]), 1 / Math.sqrt(HIDDEN)) as Tensor,
    b1: zeros([HIDDEN]),
    gate: mul(normals(child(s, 'gate'), [HIDDEN, 1]), 1 / Math.sqrt(HIDDEN)) as Tensor,
    w2: mul(normals(child(s, 'w2'), [HIDDEN, HIDDEN]), 1 / Math.sqrt(HIDDEN)) as Tensor,
    b2: zeros([HIDDEN]),
    w3: mul(normals(child(s, 'w3'), [HIDDEN, classes]), 1 / Math.sqrt(HIDDEN)) as Tensor,
    b3: zeros([classes]),
  }
}

const accuracy = (p: Params, b: GraphBatch, readout: Readout, labels: readonly number[]) => {
  const rows = toRows(logits(p, b, readout) as Tensor) as number[][]
  return rows.filter((r, i) => r.indexOf(Math.max(...r)) === labels[i]).length / labels.length
}

/** Full-batch Adam for every readout in turn, one step each per round, with a checkpoint every few steps. */
export function* readoutRun(setup: Setup): Generator<Checkpoint> {
  const labels = labelsFor(setup.task)
  const classes = setup.task === 'family' ? 3 : 2
  const yTrain = tensor(labels.slice(0, TRAIN))
  const runs = READOUTS.map((readout) => {
    const alg = trainingLoop({
      loss: (p: Params) => softmaxCrossEntropy(logits(p, TRAIN_BATCH, readout), yTrain),
      data: { y: yTrain },
      optimizer: adamRule({ stepSize: setup.rate }),
    })
    return live(alg, { params: init(classes, setup.seed) }, { stream: stream(`graph-readout/run/${setup.seed}`) })
  })
  const state = READOUTS.map(() => null as null | { params: Params; loss: number })
  const checkpoint = (step: number): Checkpoint => {
    const rec = (f: (k: number, r: Readout) => number) =>
      Object.fromEntries(READOUTS.map((r, k) => [r, f(k, r)])) as Record<Readout, number>
    return {
      step,
      loss: rec((k) => state[k]!.loss),
      train: rec((k, r) => accuracy(state[k]!.params, TRAIN_BATCH, r, labels.slice(0, TRAIN))),
      test: rec((k, r) => accuracy(state[k]!.params, TEST_BATCH, r, labels.slice(TRAIN))),
    }
  }
  for (let step = 0; step <= setup.steps; step++) {
    runs.forEach((it, k) => {
      const next = it.next()
      if (!next.done && !next.value.state.diverged)
        state[k] = { params: next.value.state.params, loss: next.value.state.loss }
    })
    if (step % EVERY === 0 || step === setup.steps) yield checkpoint(step)
  }
}

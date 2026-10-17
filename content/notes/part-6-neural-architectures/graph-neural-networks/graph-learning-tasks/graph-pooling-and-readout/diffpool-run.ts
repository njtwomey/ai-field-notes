/**
 * The DiffPool run behind the assignment figure: a one-layer graph convolution maps one-hot node identities to
 * assignment logits on a ring of three triangles, and Adam minimises DiffPool's two auxiliary losses (link prediction
 * and assignment entropy) with no task loss, so the assignment it finds is the one those losses alone favour. The
 * convolution and the training are engine calls; the coarsening is this sub-chapter's pooling candidate.
 */
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { add, eye, item, mul, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import type { Graph } from 'aifn-compute/graph'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { diffPool, graphConv } from 'aifn-compute/nn/graph'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'

/** Three triangles {0, 1, 2}, {3, 4, 5}, {6, 7, 8} joined in a ring by the edges 2-3, 5-6 and 8-0. */
export const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [0, 2],
  [3, 4],
  [4, 5],
  [3, 5],
  [6, 7],
  [7, 8],
  [6, 8],
  [2, 3],
  [5, 6],
  [8, 0],
]
export const N = 9
export const GRAPH: Graph = {
  kind: 'graph',
  nodes: N,
  edges: EDGES.map(([from, to]) => ({ from, to })),
  directed: false,
} as Graph
const A = adjacencyMatrix(GRAPH)
const X = eye(N)

/** Optimiser steps between checkpoints. */
export const EVERY = 10

export type Setup = { clusters: number; steps: number; rate: number; entropyWeight: number; seed: number }
export type Checkpoint = {
  step: number
  /** The assignment S, one row of cluster probabilities per node. */
  s: number[][]
  /** The coarsened adjacency S^T A S. */
  coarse: number[][]
  linkLoss: number
  entropy: number
}

type Params = { weight: Tensor }

/** A rank-0 value as a number. */
const num = (v: Value) => (typeof v === 'number' ? v : item(v as Tensor))

const pooled = (p: Params) => diffPool(A, X, graphConv(GRAPH, X, p.weight))

/** Adam on the link loss plus the weighted entropy, with a checkpoint every few steps. */
export function* diffPoolRun(setup: Setup): Generator<Checkpoint> {
  const loss = (p: Params) => {
    const d = pooled(p)
    return add(d.linkLoss, mul(d.entropy, setup.entropyWeight))
  }
  const alg = trainingLoop({ loss, data: { i: eye(1) }, optimizer: adamRule({ stepSize: setup.rate }) })
  const s0 = stream(`diffpool/init/${setup.seed}`)
  const start = { params: { weight: mul(normals(child(s0, 'w'), [N, setup.clusters]), 0.1) as Tensor } }
  for (const { step, state } of live(alg, start, { stream: stream('diffpool/run') })) {
    if (step % EVERY === 0 || step >= setup.steps || state.diverged) {
      const d = pooled(state.params)
      yield {
        step,
        s: toRows(d.s as Tensor) as number[][],
        coarse: toRows(d.adjacency as Tensor) as number[][],
        linkLoss: num(d.linkLoss),
        entropy: num(d.entropy),
      }
    }
    if (step >= setup.steps || state.diverged) return
  }
}

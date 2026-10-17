/**
 * The planted-partition graph shared by the oversmoothing and residual figures, and the linear readout that scores a
 * layer's node features: two communities of 20 nodes with 2-D features, 4 labelled nodes per community, and a ridge
 * regression of the ±1 labels on the features, scored on the 32 unlabelled nodes.
 */
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { dataset } from 'aifn-compute/learning/estimators'
import { linearRegression } from 'aifn-methods/learning/linear'
import { contextualBlockModel } from 'aifn-methods/data/synthetic'

/** Nodes per community. */
export const BLOCK = 20
/** The labelled nodes: the first four of each community. */
export const LABELLED = [0, 1, 2, 3, BLOCK, BLOCK + 1, BLOCK + 2, BLOCK + 3]
/** The ridge penalty of the readout, the analogue of weight decay. */
export const READOUT_L2 = 0.05

/** A planted-partition graph with node features. */
export interface Communities {
  graph: Graph
  /** Node features, one row of 2 per node. */
  x: number[][]
  /** Community of each node, 0 or 1. */
  y: number[]
  /** Share of edges inside a community. */
  homophily: number
}

/**
 * Two communities of 20 nodes: an edge joins two nodes with probability `pIn` inside a community and `pOut` across;
 * node features are $\xvec_i = \evec_{y_i} + \epsilonvec_i$ in $\reals^2$ with standard normal noise.
 *
 * @param seed The random seed.
 * @param pIn Edge probability inside a community.
 * @param pOut Edge probability across communities.
 * @returns The graph, the features and the communities.
 */
export function plantedPartition(seed: number, pIn: number, pOut: number): Communities {
  const d = contextualBlockModel(stream(seed), { sizes: [BLOCK, BLOCK], pIn, pOut, features: 2, signal: 1 })
  return {
    graph: d.graph,
    x: toRows(d.x as Tensor),
    y: toFlat(d.y as Tensor).map((v) => Math.round(v)),
    homophily: d.homophily,
  }
}

/**
 * Accuracy on the unlabelled nodes of a ridge readout fitted to the labelled nodes' ±1 labels, predicting by sign.
 *
 * @param h Node features, one row per node.
 * @param y Every node's community (only the labelled ones are fitted).
 * @param l2 The ridge penalty.
 * @returns The share of unlabelled nodes classified correctly.
 */
export function readoutAccuracy(h: readonly number[][], y: readonly number[], l2 = READOUT_L2): number {
  const train = LABELLED
  const model = linearRegression({ l2 }).fit(
    dataset(fromRows(train.map((i) => h[i])), tensor(train.map((i) => (y[i] === 1 ? 1 : -1)))),
  )
  const score = toFlat(model.decide(fromRows(h as number[][])) as Tensor)
  let right = 0
  let count = 0
  for (let i = 0; i < h.length; i++) {
    if (train.includes(i)) continue
    count++
    if (score[i] > 0 === (y[i] === 1)) right++
  }
  return count > 0 ? right / count : NaN
}

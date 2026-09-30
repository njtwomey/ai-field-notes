/**
 * The Ising model (Ising, 1925; Koller and Friedman, 2009, "Probabilistic Graphical Models", §4.4.1), part of
 * `aifn-applied/inference/lattice-models`: pairwise spins on any graph as a discrete factor graph, on a lattice
 * declared with `aifn/graph/structured`'s `latticeTemplate`, and belief propagation chosen by the graph's shape.
 */

import type { Status } from 'aifn/foundation/contracts'
import type { Algorithm } from 'aifn/foundation/trace'
import { type Graph } from 'aifn/graph'
import { latticeTemplate, shape, unroll, type GraphShape } from 'aifn/graph/structured'
import { chainSumProduct } from 'aifn/inference/exact'
import { beliefPropagationSteps, type BeliefPropagationOptions } from 'aifn/inference/message-passing'
import {
  bipartiteGraph,
  type DiscreteFactorGraph,
  type DiscreteFactor,
  discreteFactor,
  discreteFactorGraph,
} from 'aifn/inference/model'

/**
 * A pairwise Ising model on a graph as a factor graph: spins xᵢ ∈ {−1, +1} (index 0 is −1, index 1 is +1), with
 * p(x) ∝ exp(Σ_{(i,j)} J xᵢxⱼ + Σᵢ hᵢ xᵢ). `coupling` and `field` may be numbers or per-edge / per-node arrays.
 */
export function isingModel(
  graph: Graph,
  coupling: number | ArrayLike<number>,
  field: number | ArrayLike<number>,
): DiscreteFactorGraph {
  const J = (k: number) => (typeof coupling === 'number' ? coupling : coupling[k])
  const h = (i: number) => (typeof field === 'number' ? field : field[i])
  const cards = new Array<number>(graph.nodes).fill(2)
  const spin = (a: number) => (a === 0 ? -1 : 1)
  const factors: DiscreteFactor[] = []
  for (let i = 0; i < graph.nodes; i++)
    factors.push(discreteFactor([i], cards, (a) => Math.exp(h(i) * spin(a[0])), `φ${i}`))
  graph.edges.forEach((e, k) =>
    factors.push(
      discreteFactor([e.from, e.to], cards, (a) => Math.exp(J(k) * spin(a[0]) * spin(a[1])), `ψ${e.from},${e.to}`),
    ),
  )
  return discreteFactorGraph(cards, factors, graph.labels)
}

/**
 * The Ising model on a rows × cols lattice (4 neighbours; a torus with `periodic`): the structure is
 * `latticeTemplate(rows, cols)`, unrolled so that site (i, j) is node i·cols + j.
 */
export function isingLattice(
  rows: number,
  cols: number,
  coupling: number | ArrayLike<number>,
  field: number | ArrayLike<number>,
  options: { periodic?: boolean } = {},
): DiscreteFactorGraph {
  return isingModel(unroll(latticeTemplate(rows, cols, { periodic: options.periodic })), coupling, field)
}

/** The shape of an Ising model's interaction graph: `chain`, `tree`, `lattice` or `general`. */
export function isingShape(model: DiscreteFactorGraph): GraphShape {
  return shape(bipartiteGraph(model))
}

/**
 * Marginal inference on an Ising model with the engine its shape allows: forward–backward on a chain
 * (`chainSumProduct`, exact), belief propagation with the tree schedule on a tree (exact), and loopy belief propagation
 * (flooding, with `options`) otherwise (Murphy, Weiss and Jordan, 1999). Run it with `run(alg, undefined, n)`.
 */
export function isingInference(
  model: DiscreteFactorGraph,
  options: BeliefPropagationOptions = {},
): { shape: GraphShape; algorithm: Algorithm<void, Status> } {
  const graphShape = isingShape(model)
  if (graphShape === 'chain' && options.evidence === undefined && options.mode !== 'max')
    return { shape: graphShape, algorithm: chainSumProduct(model) }
  return { shape: graphShape, algorithm: beliefPropagationSteps(model, options) }
}

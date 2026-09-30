/**
 * `aifn/graph/propagation`: differentiable message passing over a graph's edges (gather → edge function → sum, mean or
 * max aggregation), the building block of graph neural networks and label propagation.
 */

export { messageEdges, propagate, type Aggregation, type EdgeFunction, type PropagateOptions } from './propagation'

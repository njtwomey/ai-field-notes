/**
 * Graph solvers for `aifn/programming`: shortest paths (Dijkstra, Bellman–Ford, Floyd–Warshall) and min-cost flow.
 * Their one definition lives in `aifn/graph`; this file re-exports them under programming's names.
 */

import { fromMatrix, type Graph } from 'aifn/graph'
import type { Tensor } from 'aifn/tensor'

export {
  bellmanFord,
  bellmanFordSteps as bellmanFordSearch,
  dijkstra,
  dijkstraSteps as dijkstraSearch,
  floydWarshall,
  floydWarshallSteps as floydWarshallSearch,
  minCostFlow,
  minCostFlowSteps as minCostFlowSearch,
  shortestPath,
  type BellmanFordState,
  type DijkstraState,
  type Edge,
  type FloydWarshallState,
  type FlowArc,
  type FlowNetwork,
  type Graph,
  type MinCostFlowState,
  type ShortestPathOptions,
  type ShortestPaths,
} from 'aifn/graph'

/**
 * A directed graph from a dense weight matrix: an edge i → j for every entry that is not `absent` (default Infinity,
 * as for a distance matrix; pass 0 to read zeros as missing edges, as `scipy.sparse.csgraph` does for dense input).
 * `fromMatrix` in `aifn/graph` with a different default.
 */
export function graphFromMatrix(weights: Tensor | readonly (readonly number[])[], absent = Infinity): Graph {
  return fromMatrix(weights, { absent })
}

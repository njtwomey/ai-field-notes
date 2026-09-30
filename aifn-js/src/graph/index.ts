/**
 * `aifn/graph`: graphs as plain data and the classical algorithms on them. Every search is a traceable `Algorithm`
 * (`aifn/trace`, named `…Steps`) whose states expose the frontier, visited set, current node and edge, parent tree
 * and times, with a convenience function that runs it to the end.
 *
 * - Representation: `Graph` (`{ nodes, edges, directed?, labels? }`), `fromEdges`, `fromAdjacency`, `fromMatrix`,
 *   `adjacency`, `neighbours`, `inDegree`, `outDegree`, `reverse`, `subgraph`, `toMatrix`, `path`. Neighbour order
 *   follows the edge list (see `adjacency`), so every traversal is reproducible.
 * - Traversal: `breadthFirstSearch`, `depthFirstSearch` (discovery and finish times, edge classes),
 *   `iterativeDeepening`.
 * - Order: `topologicalSort` (Kahn or depth-first), `kahnSteps`, `findCycle`, `isDag`.
 * - Components: `connectedComponents`, `stronglyConnectedComponents` (Tarjan, Kosaraju), `condensation`,
 *   `bipartite`.
 * - Shortest paths: `unweightedShortestPaths`, `dijkstra`, `aStar`, `bellmanFord` (negative-cycle witness),
 *   `floydWarshall`, `shortestPath`.
 * - Trees and flows: `minimumSpanningTree` (Kruskal, Prim), `maxFlow` (Edmonds–Karp, with the minimum cut),
 *   `minCostFlow`.
 * - Rooted trees (`tree.ts`): the `Tree` type shared by every tree in aifn (Huffman codes, decision trees, dendrograms,
 *   search trees, spanning trees), with constructors (`treeFromParents`, `treeFromChildren`, `treeFromNested`,
 *   `binaryTree`, `spanningTreeOf`, `spanningForestOf`), queries (`depth`, `height`, `leaves`, `ancestors`, `lca`,
 *   `subtreeSize`, `pathToRoot`), traversals (`preOrder`, `inOrder`, `postOrder`, `levelOrder`) and `mapTree` /
 *   `foldTree`. Breadth- and depth-first results carry their search forest as `trees`.
 * - Utilities for other modules: a binary heap (`createHeap`, `heapPush`, `heapPop`, …) and `unionFind`.
 */

export {
  adjacency,
  fromAdjacency,
  fromEdges,
  fromMatrix,
  inDegree,
  neighbours,
  outDegree,
  path,
  reverse,
  subgraph,
  toMatrix,
  type Arc,
  type Edge,
  type EdgeInput,
  type Graph,
  type GraphOptions,
} from './core'
export {
  createHeap,
  findRoot,
  heapCopy,
  heapPeek,
  heapPop,
  heapPush,
  heapSorted,
  unionFind,
  unionFindCopy,
  unite,
  type Heap,
  type HeapEntry,
  type UnionFind,
} from './heap'
export {
  breadthFirstSearch,
  breadthFirstSteps,
  depthFirstSearch,
  depthFirstSteps,
  iterativeDeepening,
  iterativeDeepeningSteps,
  unweightedShortestPaths,
  type BreadthFirstEvent,
  type BreadthFirstResult,
  type BreadthFirstState,
  type DepthFirstEvent,
  type DepthFirstResult,
  type DepthFirstState,
  type EdgeClass,
  type IterativeDeepeningEvent,
  type IterativeDeepeningOptions,
  type IterativeDeepeningResult,
  type IterativeDeepeningState,
  type TraversalOptions,
} from './traversal'
export { findCycle, isDag, kahnSteps, topologicalSort, type KahnState, type TopologicalOrder } from './order'
export {
  bipartite,
  condensation,
  connectedComponents,
  kosarajuSteps,
  stronglyConnectedComponents,
  tarjanSteps,
  type BipartiteResult,
  type Components,
  type KosarajuState,
  type TarjanEvent,
  type TarjanState,
} from './components'
export {
  aStar,
  aStarSteps,
  bellmanFord,
  bellmanFordSteps,
  dijkstra,
  dijkstraSteps,
  directedArcs,
  floydWarshall,
  floydWarshallSteps,
  shortestPath,
  type BellmanFordState,
  type DijkstraState,
  type DirectedArc,
  type FloydWarshallState,
  type ShortestPathOptions,
  type ShortestPaths,
} from './paths'
export {
  kruskalSteps,
  minimumSpanningTree,
  primSteps,
  sameSet,
  type KruskalState,
  type PrimCandidate,
  type PrimState,
  type SpanningEvent,
  type SpanningTree,
} from './trees'
export {
  edmondsKarpSteps,
  maxFlow,
  minCostFlow,
  minCostFlowSteps,
  type EdmondsKarpState,
  type FlowArc,
  type FlowNetwork,
  type MaxFlowOptions,
  type MaxFlowResult,
  type MinCostFlowState,
} from './flows'
export {
  ancestors,
  binaryTree,
  depth,
  depths,
  foldTree,
  height,
  inOrder,
  isLeaf,
  lca,
  leaves,
  leftChild,
  levelOrder,
  mapTree,
  pathFromRoot,
  pathToRoot,
  postOrder,
  preOrder,
  rightChild,
  spanningForestOf,
  spanningTreeOf,
  subtreeSize,
  treeFromChildren,
  treeFromNested,
  treeFromParents,
  type NestedBinaryTree,
  type NestedTree,
  type SpanningInput,
  type SpanningTreeEdge,
  type SpanningTreeNode,
  type Tree,
  type TreeEdge,
  type TreeNode,
  type TreeOptions,
} from './tree'

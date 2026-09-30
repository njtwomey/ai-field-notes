/**
 * `aifn/programming`: classical exact solvers for small problems, correct rather than state of the art. Every
 * iterative solver is a traceable `Algorithm` (`aifn/trace`) whose states expose its internals, with a convenience
 * function that runs it to the end.
 *
 * - Linear programming (scipy's `linprog` form and dual-value convention): `simplex` (two-phase tableau, Bland's or
 *   Dantzig's rule), `interiorPoint` (Mehrotra's predictor–corrector), `lpCentralPath`, `linprog`, `dualityReport`.
 * - Quadratic programming: `activeSet`, `qpInteriorPoint`, `quadprog`, `kktReport`; box-constrained `boxQP`.
 * - Integer programming: `branchAndBound` (search tree kept), `branchAndBoundTree` (the tree as an `aifn/graph`
 *   `Tree`), `milp`, `gomory` (fractional cuts).
 * - Dynamic programming: `dynamicProgram` / `dp` over tables; knapsack, LCS, edit distance, Needleman–Wunsch and
 *   Smith–Waterman alignment with tracebacks.
 * - Graphs: Dijkstra, Bellman–Ford (negative cycles), Floyd–Warshall; min-cost flow by successive shortest paths.
 * - Assignment: the Hungarian algorithm in Munkres' steps.
 */

export type { MatrixInput, VectorInput } from './dense'
export {
  dualityReport,
  type Bound,
  type DualityReport,
  type LinearProgram,
  type LinearProgramDuals,
  type StandardForm,
} from './lp'
export {
  simplex,
  simplexDuals,
  type LinearProgramMethod,
  type LinearProgramResult,
  type LinearProgramStatus,
  type LinprogOptions,
  type SimplexEvent,
  type SimplexOptions,
  type SimplexRule,
  type SimplexState,
  type SimplexStatus,
} from './simplex'
export {
  interiorPoint,
  lpCentralPath,
  type CentralPath,
  type InteriorPointOptions,
  type InteriorPointState,
} from './interior'
export { linprog } from './linprog'
export {
  activeSet,
  boxQP,
  boxQuadprog,
  kktReport,
  qpInteriorPoint,
  quadprog,
  type ActiveSetEvent,
  type ActiveSetOptions,
  type ActiveSetState,
  type BoxQPOptions,
  type BoxQPState,
  type BoxQuadraticProgram,
  type KKTReport,
  type QPInteriorPointOptions,
  type QPInteriorPointState,
  type QuadraticProgram,
  type QuadraticProgramResult,
  type QuadraticProgramStatus,
} from './qp'
export {
  branchAndBound,
  branchAndBoundTree,
  gomory,
  milp,
  type BranchAndBoundOptions,
  type BranchAndBoundState,
  type BranchNode,
  type BranchNodeStatus,
  type Cut,
  type GomoryOptions,
  type GomoryState,
  type IntegerProgramStatus,
  type MixedIntegerProgram,
  type MixedIntegerResult,
  type NodeSelection,
  type SearchTreeEdgeData,
  type SearchTreeNodeData,
} from './milp'
export {
  alignmentProgram,
  DIAGONAL,
  dp,
  dynamicProgram,
  editDistance,
  editDistanceProgram,
  knapsack,
  knapsackProgram,
  lcs,
  lcsProgram,
  LEFT,
  needlemanWunsch,
  smithWaterman,
  STOP,
  unboundedKnapsack,
  unboundedKnapsackProgram,
  UP,
  type AlignmentResult,
  type AlignmentScoring,
  type DynamicProgram,
  type DynamicProgramState,
  type EditCosts,
  type EditDistanceResult,
  type EditOperation,
  type KnapsackResult,
  type LCSResult,
  type Sequence,
} from './dp'
export {
  bellmanFord,
  bellmanFordSearch,
  dijkstra,
  dijkstraSearch,
  floydWarshall,
  floydWarshallSearch,
  graphFromMatrix,
  minCostFlow,
  minCostFlowSearch,
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
} from './graphs'
export {
  hungarian,
  hungarianSteps,
  type AssignmentResult,
  type HungarianOptions,
  type HungarianPhase,
  type HungarianState,
} from './assignment'

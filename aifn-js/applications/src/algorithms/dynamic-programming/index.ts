/**
 * `aifn-applied/algorithms/dynamic-programming`: dynamic programmes: 0/1 and unbounded knapsacks, longest common
 * subsequence, edit distance, and Needleman–Wunsch and Smith–Waterman alignment, on the `dp` engine of
 * `aifn/optim/programming`.
 */
export {
  knapsackProgram,
  type KnapsackResult,
  knapsack,
  unboundedKnapsackProgram,
  unboundedKnapsack,
  type Sequence,
  DIAGONAL,
  UP,
  LEFT,
  STOP,
  lcsProgram,
  type LCSResult,
  lcs,
  type EditCosts,
  editDistanceProgram,
  type EditOperation,
  type EditDistanceResult,
  editDistance,
  type AlignmentScoring,
  alignmentProgram,
  type AlignmentResult,
  needlemanWunsch,
  smithWaterman,
} from './problems'
export { dynamicProgrammingFunctions } from './registry'

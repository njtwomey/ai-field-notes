import type { Specimen } from '../../specimen'
import {
  ActiveSetSpecimen,
  AlignmentSpecimen,
  BranchAndBoundSpecimen,
  GomorySpecimen,
  HungarianSpecimen,
  InteriorPointSpecimen,
  KnapsackSpecimen,
  ProgramViewSpecimen,
  SimplexSpecimen,
} from './_programming/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim/programming',
    title: 'Simplex pivots on a 2-D LP',
    description:
      'The two-phase simplex method walks the vertices of a feasible polygon; the tableau, entering column, leaving row and ratio test are shown at every pivot.',
    tags: ['simplex', 'tableau', 'Bland', 'phase 1', 'linear programming'],
    render: () => <SimplexSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Interior point and the central path',
    description:
      "Mehrotra's predictor–corrector iterates against the exact central path x(μ), from the analytic centre to the optimal vertex.",
    tags: ['interior point', 'Mehrotra', 'central path', 'barrier', 'linear programming'],
    render: () => <InteriorPointSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Branch-and-bound tree',
    description:
      'A small integer program solved by branch and bound: the search tree with each node branched, integral or pruned, and the node’s bounds over the LP relaxation.',
    tags: ['branch and bound', 'integer programming', 'MILP', 'search tree'],
    render: () => <BranchAndBoundSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Gomory cutting planes',
    description:
      'Fractional cuts read from the optimal tableau shrink the LP relaxation until its optimum is integral.',
    tags: ['Gomory', 'cutting planes', 'dual simplex', 'integer programming'],
    render: () => <GomorySpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Knapsack table',
    description: 'The 0/1 knapsack dynamic program filled row by row, with the traceback of the chosen items.',
    tags: ['knapsack', 'dynamic programming', 'traceback'],
    render: () => <KnapsackSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Alignment grid',
    description: 'Needleman–Wunsch and Smith–Waterman score tables with the traceback path and the aligned sequences.',
    tags: ['alignment', 'Needleman–Wunsch', 'Smith–Waterman', 'edit distance', 'dynamic programming'],
    render: () => <AlignmentSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Hungarian algorithm, step by step',
    description:
      "Munkres' steps on a random cost matrix: reductions, starred and primed zeros, covers and augmenting paths.",
    tags: ['Hungarian', 'Munkres', 'assignment'],
    render: () => <HungarianSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'Active-set and interior-point QP',
    description:
      'A convex QP over a polygon: the active-set path along faces and the interior-point path through the interior.',
    tags: ['quadratic programming', 'active set', 'KKT', 'interior point'],
    render: () => <ActiveSetSpecimen />,
  },
  {
    module: 'optim/programming',
    title: 'ProgramView: LP and QP results',
    description:
      'The generic result view: variables, objective, duals (scipy’s convention), slacks, active constraints and the optimality checks, including infeasible and unbounded outcomes.',
    tags: ['ProgramView', 'duality', 'KKT', 'complementary slackness', 'linprog', 'quadprog'],
    render: () => <ProgramViewSpecimen />,
  },
]

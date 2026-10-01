/**
 * The algorithms of `aifn/optim/programming`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as assignment from './assignment'
import * as dp from './dp'
import * as interior from './interior'
import * as milp from './milp'
import * as qp from './qp'
import * as simplex from './simplex'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/programming')

algorithm(
  {
    key: 'simplex',
    name: 'Simplex method',
    problem: 'linear-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'terminated'] },
    notes: ['linear-programming'],
  },
  simplex.simplex,
)
algorithm(
  {
    key: 'linearInteriorPoint',
    name: 'Interior point (LP)',
    problem: 'linear-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-programming'],
    cite: ['karmarkar1984'],
  },
  interior.linearInteriorPoint,
)
algorithm(
  {
    key: 'activeSet',
    name: 'Active set (QP)',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
    cite: ['nocedal2006'],
  },
  qp.activeSet,
)
algorithm(
  {
    key: 'quadraticInteriorPoint',
    name: 'Interior point (QP)',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
    cite: ['nocedal2006'],
  },
  qp.quadraticInteriorPoint,
)
algorithm(
  {
    key: 'boxQuadraticProgram',
    name: 'Box-constrained QP',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'objective', grad: 'grad', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['quadratic-programming'],
  },
  qp.boxQuadraticProgram,
)
algorithm(
  {
    key: 'branchAndBound',
    name: 'Branch and bound',
    problem: 'integer-program',
    state: { iterate: 'incumbent', objective: 'incumbentValue', flags: ['converged', 'terminated'] },
  },
  milp.branchAndBound,
)
algorithm(
  {
    key: 'gomory',
    name: 'Gomory cutting planes',
    problem: 'integer-program',
    state: { iterate: 'x', objective: 'objective', flags: ['converged', 'terminated'] },
  },
  milp.gomory,
)
algorithm(
  {
    key: 'hungarianSteps',
    name: 'Hungarian algorithm',
    problem: 'assignment',
    state: { flags: ['converged'] },
    cite: ['kuhn1955'],
  },
  assignment.hungarianSteps,
)
algorithm(
  {
    key: 'dynamicProgram',
    name: 'Dynamic programming',
    problem: 'dynamic-program',
    state: { iterate: 'table', flags: ['converged'] },
    notes: ['bellman-equations'],
    cite: ['bellman1957'],
  },
  dp.dynamicProgram,
)

/** Every algorithm of the module, keyed by factory name. */
export const programmingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', assignment, dp, interior, milp, qp, simplex) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/**
 * The algorithms of `aifn/optim/proximal`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as proximal from './proximal'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/proximal')

algorithm(
  {
    key: 'proximalGradient',
    name: 'Proximal gradient',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.proximalGradient,
)
algorithm(
  {
    key: 'ista',
    name: 'ISTA',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods', 'lasso'],
    cite: ['beck2009'],
  },
  proximal.ista,
)
algorithm(
  {
    key: 'fista',
    name: 'FISTA',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods', 'accelerated-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.fista,
)
algorithm(
  {
    key: 'projectedGradient',
    name: 'Projected gradient',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'gradY', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['proximal-gradient-methods'],
    cite: ['beck2009'],
  },
  proximal.projectedGradient,
)

/** Every algorithm of the module, keyed by factory name. */
export const proximalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', proximal) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

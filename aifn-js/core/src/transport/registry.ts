/**
 * The algorithms of `aifn/transport`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as discrete from './discrete'
import * as gromov from './gromov'

const algorithm = definer<AlgorithmInfo>('algorithm', 'transport')

algorithm(
  {
    key: 'sinkhornSteps',
    name: 'Sinkhorn',
    problem: 'transport',
    state: { iterate: 'plan', objective: 'value', flags: ['converged', 'diverged'] },
    notes: ['entropic-regularisation-and-sinkhorn', 'optimal-transport'],
    cite: ['cuturi2013'],
  },
  discrete.sinkhornSteps,
)
algorithm(
  {
    key: 'gromovWassersteinSteps',
    name: 'Gromov–Wasserstein',
    problem: 'transport',
    state: { iterate: 'plan', objective: 'loss', flags: ['converged', 'diverged'] },
    notes: ['optimal-transport'],
  },
  gromov.gromovWassersteinSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const transportAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', discrete, gromov) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

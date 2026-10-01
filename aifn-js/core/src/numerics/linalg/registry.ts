/**
 * The algorithms of `aifn/numerics/linalg`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as riccati from './riccati'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/linalg')

algorithm(
  {
    key: 'kleinmanIteration',
    name: 'Kleinman iteration',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.kleinmanIteration,
)
algorithm(
  {
    key: 'riccatiMatrixSign',
    name: 'Matrix sign function',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiMatrixSign,
)
algorithm(
  {
    key: 'riccatiRecursion',
    name: 'Riccati recursion',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiRecursion,
)
algorithm(
  {
    key: 'riccatiDoubling',
    name: 'Structured doubling',
    problem: 'riccati',
    state: { iterate: 'P', objective: 'residual', flags: ['converged', 'diverged', 'terminated'] },
    notes: ['linear-quadratic-regulator'],
  },
  riccati.riccatiDoubling,
)

/** Every algorithm of the module, keyed by factory name. */
export const linalgAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', riccati) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

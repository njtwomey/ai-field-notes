/**
 * The algorithms of `aifn/numerics/quadrature`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as adaptive from './adaptive'
import * as multivariate from './multivariate'
import * as rules from './rules'

const algorithm = definer<AlgorithmInfo>('algorithm', 'numerics/quadrature')

algorithm(
  {
    key: 'adaptiveSimpson',
    stability: 'stable',
    name: 'Adaptive Simpson',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['numerical-integration'],
  },
  adaptive.adaptiveSimpson,
)
algorithm(
  {
    key: 'gaussKronrod',
    stability: 'stable',
    name: 'Adaptive Gauss–Kronrod',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['numerical-integration'],
  },
  adaptive.gaussKronrod,
)
algorithm(
  {
    key: 'romberg',
    stability: 'stable',
    name: 'Romberg',
    problem: 'integral',
    state: { iterate: 'value', objective: 'error', flags: ['converged', 'diverged'] },
    notes: ['numerical-integration'],
  },
  rules.romberg,
)
algorithm(
  {
    key: 'monteCarlo',
    stability: 'stable',
    name: 'Monte Carlo integration',
    problem: 'integral',
    state: { iterate: 'value', objective: 'standardError', flags: ['diverged'] },
    random: true,
    notes: ['monte-carlo-integration'],
  },
  multivariate.monteCarlo,
)

/** Every algorithm of the module, keyed by factory name. */
export const quadratureAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive, multivariate, rules) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

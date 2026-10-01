/**
 * The algorithms of `aifn/dynamics/sde`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as integrators from './integrators'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/sde')

algorithm(
  {
    key: 'eulerMaruyama',
    name: 'Euler–Maruyama',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['euler-maruyama-method', 'stochastic-differential-equations'],
    cite: ['maruyama1955', 'kloeden1992'],
  },
  integrators.eulerMaruyama,
)
algorithm(
  {
    key: 'milstein',
    name: 'Milstein',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-differential-equations'],
    cite: ['kloeden1992'],
  },
  integrators.milstein,
)
algorithm(
  {
    key: 'stochasticRungeKutta',
    name: 'Stochastic Runge–Kutta',
    problem: 'sde',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-differential-equations'],
    cite: ['kloeden1992'],
  },
  integrators.stochasticRungeKutta,
)

/** Every algorithm of the module, keyed by factory name. */
export const sdeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', integrators) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

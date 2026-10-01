/**
 * The algorithms of `aifn/inference/filtering`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as changepoint from './changepoint'
import * as kalman from './kalman'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/filtering')

algorithm(
  {
    key: 'bocpd',
    name: 'Bayesian online changepoint detection',
    problem: 'sequence',
    state: { iterate: 'logPosterior', objective: 'logEvidence', flags: ['terminated'] },
    notes: ['bayesian-online-changepoint-detection'],
    cite: ['adams2007'],
  },
  changepoint.bocpd,
)

algorithm(
  {
    key: 'kalmanFilterSteps',
    name: 'Kalman filter',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logLikelihood', flags: ['terminated'] },
    notes: ['kalman-filter'],
    cite: ['kalman1960'],
  },
  kalman.kalmanFilterSteps,
)

algorithm(
  {
    key: 'rtsSmootherSteps',
    name: 'Rauch–Tung–Striebel smoother',
    problem: 'sequence',
    state: { iterate: 'mean', flags: ['terminated'] },
    notes: ['kalman-smoother'],
    cite: ['rauch1965'],
  },
  kalman.rtsSmootherSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const filteringAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', changepoint, kalman) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

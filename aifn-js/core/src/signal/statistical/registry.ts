/**
 * The algorithms of `aifn/signal/statistical`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, and the `Status` flags it sets), so a generic trace view picks default
 * series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as adaptive from './adaptive'

const algorithm = definer<AlgorithmInfo>('algorithm', 'signal/statistical')

algorithm(
  {
    key: 'lms',
    stability: 'stable',
    name: 'Least mean squares (LMS)',
    summary: 'An adaptive FIR filter that takes one stochastic-gradient step on the squared error per sample.',
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['least-mean-squares-filter'],
    cite: ['widrow1976', 'sayed2008'],
  },
  adaptive.lms,
)
algorithm(
  {
    key: 'nlms',
    stability: 'stable',
    name: 'Normalised LMS',
    summary:
      "LMS with each step divided by the regressor's energy, so the step size does not depend on the input scale.",
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['least-mean-squares-filter'],
    cite: ['sayed2008'],
  },
  adaptive.nlms,
)
algorithm(
  {
    key: 'rls',
    stability: 'stable',
    name: 'Recursive least squares (RLS)',
    summary:
      'An adaptive FIR filter that keeps the exact exponentially weighted least-squares taps, updated per sample.',
    problem: 'signal',
    state: { iterate: 'w', objective: 'squaredError', flags: ['terminated'] },
    notes: ['recursive-least-squares-filter'],
    cite: ['sayed2008'],
  },
  adaptive.rls,
)

/** Every algorithm of the module, keyed by factory name. */
export const statisticalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

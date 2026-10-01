/**
 * The algorithms of `aifn/systems`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as responses from './responses'

const algorithm = definer<AlgorithmInfo>('algorithm', 'systems')

algorithm(
  {
    key: 'simulate',
    name: 'LTI simulation',
    summary: 'Steps a discrete LTI system x[k+1] = A x[k] + B u[k] through an input sequence.',
    problem: 'lti-system',
    state: { iterate: 'x', flags: ['diverged', 'terminated'] },
  },
  responses.simulate,
)

/** Every algorithm of the module, keyed by factory name. */
export const systemsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', responses) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

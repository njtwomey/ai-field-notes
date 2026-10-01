/**
 * The algorithms of `aifn/inference/message-passing`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as bp from './bp'
import * as gaussianBp from './gaussianBp'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/message-passing')

algorithm(
  {
    key: 'beliefPropagationSteps',
    name: 'Belief propagation',
    problem: 'factor-graph',
    state: { iterate: 'beliefs', objective: 'change', flags: ['converged'] },
    notes: ['belief-propagation', 'loopy-belief-propagation'],
    cite: ['pearl1988'],
  },
  bp.beliefPropagationSteps,
)
algorithm(
  {
    key: 'gaussianBeliefPropagationSteps',
    name: 'Gaussian belief propagation',
    problem: 'gaussian-model',
    state: { iterate: 'means', objective: 'change', flags: ['converged', 'diverged'] },
    notes: ['belief-propagation'],
  },
  gaussianBp.gaussianBeliefPropagationSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const messagePassingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', bp, gaussianBp) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

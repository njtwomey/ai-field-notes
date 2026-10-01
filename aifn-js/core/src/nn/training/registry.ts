/**
 * The algorithms of `aifn/nn/training`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as train from './train'

const algorithm = definer<AlgorithmInfo>('algorithm', 'nn/training')

algorithm(
  {
    key: 'trainingLoop',
    name: 'Training loop',
    summary: 'Minibatch training of a network: forward, loss, gradients and an optimiser update per step.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: ['backpropagation'],
  },
  train.trainingLoop,
)

/** Every algorithm of the module, keyed by factory name. */
export const trainingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', train) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

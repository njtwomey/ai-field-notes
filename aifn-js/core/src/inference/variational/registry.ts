/**
 * The algorithms of `aifn/inference/variational`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as bbvi from './bbvi'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/variational')

algorithm(
  {
    key: 'bbvi',
    name: 'Black-box variational inference',
    problem: 'log-density',
    state: { iterate: 'mean', objective: 'elbo', grad: 'grad', flags: ['diverged'] },
    random: true,
    glossary: 'vi',
    notes: ['black-box-variational-inference', 'variational-inference'],
    cite: ['ranganath2014'],
  },
  bbvi.bbvi,
)

/** Every algorithm of the module, keyed by factory name. */
export const variationalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', bbvi) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

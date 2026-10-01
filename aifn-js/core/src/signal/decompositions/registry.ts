/**
 * The algorithms of `aifn/signal/decompositions`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as emd from './emd'
import * as vmd from './vmd'

const algorithm = definer<AlgorithmInfo>('algorithm', 'signal/decompositions')

algorithm(
  {
    key: 'siftSteps',
    name: 'EMD sifting',
    summary: 'The sifting iterations that extract one intrinsic mode function.',
    problem: 'signal',
    state: { iterate: 'h', flags: ['converged', 'terminated'] },
    notes: ['empirical-mode-decomposition'],
    cite: ['huang1998'],
  },
  emd.siftSteps,
)

algorithm(
  {
    key: 'vmdSteps',
    name: 'Variational mode decomposition',
    summary: 'ADMM sweeps that fit K narrow-band modes and their centre frequencies together, in the Fourier domain.',
    problem: 'signal',
    state: { iterate: 'omega', objective: 'change', flags: ['converged'] },
    notes: ['variational-mode-decomposition'],
    cite: ['dragomiretskiy2014'],
  },
  vmd.vmdSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const decompositionsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', emd, vmd) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

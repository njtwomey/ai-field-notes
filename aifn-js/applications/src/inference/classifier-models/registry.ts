/**
 * The registry of `aifn-applied/inference/classifier-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as bpm from './bayesPointMachine'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/classifier-models')
const fn = definer<FunctionInfo>('function', 'inference/classifier-models')
const notes = ['bayes-point-machine', 'expectation-propagation-probit-regression', 'expectation-propagation']

algorithm(
  {
    key: 'bayesPointMachine',
    name: 'Bayes point machine (EP)',
    summary: 'A Gaussian posterior over linear-classifier weights by expectation propagation, one site per example.',
    problem: 'factor-graph',
    state: { iterate: 'mean', flags: ['converged'] },
    notes,
    cite: ['herbrich2001', 'minka2001'],
  },
  bpm.bayesPointMachine,
)
fn(
  { key: 'bayesPointMachinePredict', name: 'Bayes point machine prediction', role: 'inference', notes },
  bpm.bayesPointMachinePredict,
)

/** The algorithms of the module. */
export const classifierModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  bpm,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const classifierModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  bpm,
) as Table<FunctionInfo>

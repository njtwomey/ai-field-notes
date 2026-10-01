/**
 * The registry of `aifn/learning/calibration`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as isotonic from './isotonic'

const NOTES = ['isotonic-regression', 'isotonic-calibration']

definer<AlgorithmInfo>('algorithm', 'learning/calibration')(
  {
    key: 'poolAdjacentViolatorsSteps',
    name: 'Pool adjacent violators',
    summary: 'Add points one at a time and pool the last two blocks while they violate the order.',
    problem: 'least-squares',
    state: { iterate: 'fit', objective: 'sse', flags: [] },
    notes: NOTES,
    cite: ['ayer1955', 'best1990'],
  },
  isotonic.poolAdjacentViolatorsSteps,
)
definer<FunctionInfo>('function', 'learning/calibration')(
  {
    key: 'isotonicRegression',
    name: 'Isotonic regression',
    summary: 'The monotone least-squares fit, with ties in x pooled first.',
    role: 'fit',
    notes: NOTES,
    cite: ['ayer1955', 'zadrozny2002'],
  },
  isotonic.isotonicRegression,
)

/** The algorithms of the module, keyed by factory name. */
export const calibrationAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', isotonic) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const calibrationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', isotonic) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

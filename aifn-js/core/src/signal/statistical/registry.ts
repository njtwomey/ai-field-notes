/**
 * The algorithms of `aifn/signal/statistical`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, and the `Status` flags it sets), so a generic trace view picks default
 * series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as autoregression from './autoregression'
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

const fn = definer<FunctionInfo>('function', 'signal/statistical')
const AR = ['parametric-spectral-estimation', 'linear-prediction', 'autoregressive-model']

fn(
  {
    key: 'yuleWalker',
    name: 'Yule–Walker AR estimate',
    summary: 'AR coefficients from the sample autocorrelation by the Levinson–Durbin recursion.',
    role: 'estimator',
    notes: AR,
    cite: ['yule1927', 'walker1931'],
  },
  autoregression.yuleWalker,
)
fn(
  {
    key: 'burg',
    name: "Burg's AR estimate",
    summary: 'AR coefficients minimising forward and backward prediction errors, one reflection coefficient per order.',
    role: 'estimator',
    notes: AR,
  },
  autoregression.burg,
)

/** The functions of the module, keyed by name. */
export const statisticalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', autoregression) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >

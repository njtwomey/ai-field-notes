/**
 * The algorithms of `aifn/inference/expectation-propagation`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as ep from './ep'
import * as model from './model'
import * as multivariate from './multivariate'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/expectation-propagation')

algorithm(
  {
    key: 'expectationPropagation',
    name: 'Expectation propagation',
    problem: 'gaussian-model',
    state: { iterate: 'posterior', flags: ['converged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001'],
  },
  ep.expectationPropagation,
)
algorithm(
  {
    key: 'assumedDensityFiltering',
    name: 'Assumed density filtering',
    problem: 'gaussian-model',
    state: { iterate: 'posterior', objective: 'logEvidence', flags: ['converged'] },
    notes: ['assumed-density-filtering'],
    cite: ['minka2001'],
  },
  ep.assumedDensityFiltering,
)

algorithm(
  {
    key: 'multivariateExpectationPropagation',
    name: 'Expectation propagation (multivariate, rank-one sites)',
    problem: 'gaussian-model',
    state: { iterate: 'mean', objective: 'logEvidence', flags: ['converged', 'diverged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001'],
  },
  multivariate.multivariateExpectationPropagation,
)

algorithm(
  {
    key: 'modelExpectationPropagation',
    name: 'Expectation propagation over a linear-Gaussian model',
    problem: 'gaussian-model',
    state: { iterate: 'means', flags: ['converged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001', 'herbrich2006'],
  },
  model.modelExpectationPropagation,
)

/** Every algorithm of the module, keyed by factory name. */
export const expectationPropagationAlgorithms: Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
> = entries<AlgorithmInfo>('algorithm', ep, model, multivariate) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

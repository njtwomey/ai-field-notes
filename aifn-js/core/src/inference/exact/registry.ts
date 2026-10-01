/**
 * The algorithms of `aifn/inference/exact`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as chain from './chain'
import * as exact from './exact'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/exact')

algorithm(
  {
    key: 'enumerationSteps',
    name: 'Enumeration',
    summary: 'Exact inference by enumerating every assignment.',
    problem: 'factor-graph',
    state: { iterate: 'logMarginals', objective: 'logZ', flags: [] },
    notes: ['variable-elimination'],
  },
  exact.enumerationSteps,
)
algorithm(
  {
    key: 'variableEliminationSteps',
    name: 'Variable elimination',
    problem: 'factor-graph',
    state: { flags: [] },
    notes: ['variable-elimination'],
    cite: ['zhang1994'],
  },
  exact.variableEliminationSteps,
)
algorithm(
  {
    key: 'forwardBackwardSteps',
    name: 'Forward–backward',
    problem: 'chain',
    state: { iterate: 'marginals', objective: 'logLikelihood', flags: [] },
    notes: ['hidden-markov-model'],
    cite: ['baum1970'],
  },
  chain.forwardBackwardSteps,
)
algorithm(
  {
    key: 'viterbiSteps',
    name: 'Viterbi',
    problem: 'chain',
    state: { iterate: 'path', flags: [] },
    glossary: 'viterbi',
    notes: ['hidden-markov-model'],
    cite: ['viterbi1967'],
  },
  chain.viterbiSteps,
)
algorithm(
  {
    key: 'chainSumProduct',
    name: 'Chain sum–product',
    summary: 'Sum–product (or max–product) on a chain-shaped factor graph.',
    problem: 'factor-graph',
    state: { iterate: 'marginals', objective: 'logZ', flags: [] },
    notes: ['belief-propagation', 'factor-graph'],
  },
  chain.chainSumProduct,
)

/** Every algorithm of the module, keyed by factory name. */
export const exactAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', chain, exact) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

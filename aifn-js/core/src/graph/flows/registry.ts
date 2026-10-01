/**
 * The algorithms of `aifn/graph/flows`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as flows from './flows'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/flows')

algorithm(
  {
    key: 'edmondsKarpSteps',
    name: 'Edmonds–Karp',
    summary: 'Maximum flow by shortest augmenting paths.',
    problem: 'flow-network',
    state: { iterate: 'flow', objective: 'value', flags: [] },
  },
  flows.edmondsKarpSteps,
)
algorithm(
  {
    key: 'minCostFlowSteps',
    name: 'Minimum-cost flow',
    summary: 'Minimum-cost flow by successive shortest paths with potentials.',
    problem: 'flow-network',
    state: { iterate: 'flow', objective: 'cost', flags: [] },
  },
  flows.minCostFlowSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const flowsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', flows) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

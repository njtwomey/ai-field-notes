/**
 * The algorithms of `aifn/graph/traversal`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as components from './components'
import * as order from './order'
import * as traversal from './traversal'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/traversal')

algorithm(
  { key: 'breadthFirstSteps', name: 'Breadth-first search', problem: 'graph', state: { iterate: 'order', flags: [] } },
  traversal.breadthFirstSteps,
)
algorithm(
  { key: 'depthFirstSteps', name: 'Depth-first search', problem: 'graph', state: { iterate: 'preorder', flags: [] } },
  traversal.depthFirstSteps,
)
algorithm(
  { key: 'iterativeDeepeningSteps', name: 'Iterative deepening', problem: 'graph', state: { flags: [] } },
  traversal.iterativeDeepeningSteps,
)
algorithm(
  { key: 'kahnSteps', name: 'Kahn topological sort', problem: 'graph', state: { iterate: 'order', flags: [] } },
  order.kahnSteps,
)
algorithm(
  {
    key: 'tarjanSteps',
    name: 'Tarjan strongly connected components',
    problem: 'graph',
    state: { iterate: 'component', flags: [] },
  },
  components.tarjanSteps,
)
algorithm(
  {
    key: 'kosarajuSteps',
    name: 'Kosaraju strongly connected components',
    problem: 'graph',
    state: { iterate: 'component', flags: [] },
  },
  components.kosarajuSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const traversalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', components, order, traversal) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

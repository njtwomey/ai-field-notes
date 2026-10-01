/**
 * The algorithms of `aifn/graph/shortest-paths`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as paths from './paths'

const algorithm = definer<AlgorithmInfo>('algorithm', 'graph/shortest-paths')

algorithm(
  { key: 'dijkstraSteps', name: 'Dijkstra', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.dijkstraSteps,
)
algorithm(
  { key: 'aStarSteps', name: 'A*', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.aStarSteps,
)
algorithm(
  { key: 'bellmanFordSteps', name: 'Bellman–Ford', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.bellmanFordSteps,
)
algorithm(
  { key: 'floydWarshallSteps', name: 'Floyd–Warshall', problem: 'graph', state: { iterate: 'distance', flags: [] } },
  paths.floydWarshallSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const shortestPathsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', paths) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

import type { Specimen } from '../specimen'
import { MazeSpecimen } from './_graph/paths'
import { KahnSpecimen, SearchSpecimen, SearchTreesSpecimen, TarjanSpecimen } from './_graph/traversal'
import { MaxFlowSpecimen, SpanningTreeSpecimen } from './_graph/trees'

export const specimens: Specimen[] = [
  {
    module: 'graph',
    title: 'Breadth-first against depth-first search',
    description:
      'One graph searched both ways, edge by edge: the queue against the stack, depths against discovery/finish times, the depth-first edge classes (tree, back, forward, cross), and the two search trees drawn as trees.',
    tags: ['BFS', 'DFS', 'traversal', 'edge classification', 'queue', 'stack'],
    render: () => (
      <>
        <SearchSpecimen />
        <SearchTreesSpecimen />
      </>
    ),
  },
  {
    module: 'graph',
    title: "Kahn's topological sort",
    description:
      'Getting dressed (CLRS 22.7): nodes leave in topological order as their remaining in-degree reaches zero.',
    tags: ['topological sort', 'Kahn', 'DAG', 'in-degree'],
    render: () => <KahnSpecimen />,
  },
  {
    module: 'graph',
    title: "Tarjan's strongly connected components",
    description:
      'CLRS 22.9 stepped through Tarjan’s algorithm: discovery indices, low-links, Tarjan’s stack and components popped off it.',
    tags: ['strongly connected components', 'Tarjan', 'low-link', 'DFS'],
    render: () => <TarjanSpecimen />,
  },
  {
    module: 'graph',
    title: 'Dijkstra against A* on a maze',
    description:
      'Shortest paths on a grid maze from a binary-heap priority queue: A* with a Manhattan heuristic expands fewer cells than Dijkstra and finds the same length.',
    tags: ['Dijkstra', 'A*', 'shortest path', 'priority queue', 'heuristic', 'maze'],
    render: () => <MazeSpecimen />,
  },
  {
    module: 'graph',
    title: 'Kruskal against Prim',
    description:
      'Two ways to a minimum spanning tree of CLRS 23.1: edges by weight with union–find, or one tree grown from a heap of leaving edges.',
    tags: ['minimum spanning tree', 'Kruskal', 'Prim', 'union–find', 'heap'],
    render: () => <SpanningTreeSpecimen />,
  },
  {
    module: 'graph',
    title: 'Edmonds–Karp maximum flow and minimum cut',
    description:
      'Shortest augmenting paths on CLRS 26.1 until the sink is out of reach; the last search’s reach gives a minimum cut of capacity 23.',
    tags: ['maximum flow', 'minimum cut', 'Edmonds–Karp', 'residual network'],
    render: () => <MaxFlowSpecimen />,
  },
]

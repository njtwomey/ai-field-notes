import type { Specimen } from '../../specimen'
import { KahnSpecimen, SearchSpecimen, SearchTreesSpecimen, TarjanSpecimen } from './_traversal/traversal'

export const specimens: Specimen[] = [
  {
    module: 'graph/traversal',
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
    module: 'graph/traversal',
    title: "Kahn's topological sort",
    description:
      'Getting dressed (CLRS 22.7): nodes leave in topological order as their remaining in-degree reaches zero.',
    tags: ['topological sort', 'Kahn', 'DAG', 'in-degree'],
    render: () => <KahnSpecimen />,
  },
  {
    module: 'graph/traversal',
    title: "Tarjan's strongly connected components",
    description:
      'CLRS 22.9 stepped through Tarjan’s algorithm: discovery indices, low-links, Tarjan’s stack and components popped off it.',
    tags: ['strongly connected components', 'Tarjan', 'low-link', 'DFS'],
    render: () => <TarjanSpecimen />,
  },
]

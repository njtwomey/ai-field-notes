import type { Specimen } from '@lab/specimen'
import { KahnSpecimen, SearchSpecimen, SearchTreesSpecimen, TarjanSpecimen } from './_traversal/traversal'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/graphs-and-combinatorics',
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
    module: 'part-1-mathematical-foundations/mathematics/graphs-and-combinatorics',
    title: "Kahn's topological sort",
    description:
      'Getting dressed (CLRS 22.7): nodes leave in topological order as their remaining in-degree reaches zero.',
    tags: ['topological sort', 'Kahn', 'DAG', 'in-degree'],
    render: () => <KahnSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/graphs-and-combinatorics',
    title: "Tarjan's strongly connected components",
    description:
      'CLRS 22.9 stepped through Tarjan’s algorithm: discovery indices, low-links, Tarjan’s stack and components popped off it.',
    tags: ['strongly connected components', 'Tarjan', 'low-link', 'DFS'],
    render: () => <TarjanSpecimen />,
  },
]

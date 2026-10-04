import type { Specimen } from '@lab/specimen'
import { SpanningTreeSpecimen } from './_spanning-trees/trees'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/graphs-and-combinatorics',
    title: 'Kruskal against Prim',
    description:
      'Two ways to a minimum spanning tree of CLRS 23.1: edges by weight with union–find, or one tree grown from a heap of leaving edges.',
    tags: ['minimum spanning tree', 'Kruskal', 'Prim', 'union–find', 'heap'],
    render: () => <SpanningTreeSpecimen />,
  },
]

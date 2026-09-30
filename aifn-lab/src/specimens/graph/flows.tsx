import type { Specimen } from '../../specimen'
import { MaxFlowSpecimen } from './_flows/flows'

export const specimens: Specimen[] = [
  {
    module: 'graph/flows',
    title: 'Edmonds–Karp maximum flow and minimum cut',
    description:
      'Shortest augmenting paths on CLRS 26.1 until the sink is out of reach; the last search’s reach gives a minimum cut of capacity 23.',
    tags: ['maximum flow', 'minimum cut', 'Edmonds–Karp', 'residual network'],
    render: () => <MaxFlowSpecimen />,
  },
]

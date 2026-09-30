import type { Specimen } from '../../../../specimen'
import { CliffSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/decisions/reinforcement-learning/learning',
    title: 'Q-learning against SARSA on the cliff',
    description: 'Greedy paths and learning curves of off- and on-policy TD control on cliff walking.',
    tags: ['Q-learning', 'SARSA', 'cliff walking', 'TD control'],
    render: () => <CliffSpecimen />,
  },
]

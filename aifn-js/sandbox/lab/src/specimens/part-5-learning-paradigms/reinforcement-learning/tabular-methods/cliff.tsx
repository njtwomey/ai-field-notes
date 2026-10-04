import type { Specimen } from '@lab/specimen'
import { CliffSpecimen } from '../_rl/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/reinforcement-learning/tabular-methods',
    title: 'Q-learning against SARSA on the cliff',
    description: 'Greedy paths and learning curves of off- and on-policy TD control on cliff walking.',
    tags: ['Q-learning', 'SARSA', 'cliff walking', 'TD control'],
    render: () => <CliffSpecimen />,
  },
]

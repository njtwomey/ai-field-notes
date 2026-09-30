import type { Specimen } from '../specimen'
import { CliffSpecimen, PlanningSpecimen } from './_rl/figures'

export const specimens: Specimen[] = [
  {
    module: 'rl',
    title: 'Value and policy iteration on a grid',
    description: 'The value table and greedy-policy arrows sweep by sweep on the gridworld, a maze and FrozenLake.',
    tags: ['value iteration', 'policy iteration', 'Bellman', 'gridworld', 'FrozenLake', 'trace'],
    render: () => <PlanningSpecimen />,
  },
  {
    module: 'rl',
    title: 'Q-learning against SARSA on the cliff',
    description: 'Greedy paths and learning curves of off- and on-policy TD control on cliff walking.',
    tags: ['Q-learning', 'SARSA', 'cliff walking', 'TD control'],
    render: () => <CliffSpecimen />,
  },
]

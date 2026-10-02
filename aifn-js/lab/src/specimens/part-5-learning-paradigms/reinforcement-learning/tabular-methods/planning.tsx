import type { Specimen } from '@lab/specimen'
import { PlanningSpecimen } from '../_rl/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/reinforcement-learning/tabular-methods',
    title: 'Value and policy iteration on a grid',
    description: 'The value table and greedy-policy arrows sweep by sweep on the gridworld, a maze and FrozenLake.',
    tags: ['value iteration', 'policy iteration', 'Bellman', 'gridworld', 'FrozenLake', 'trace'],
    render: () => <PlanningSpecimen />,
  },
]

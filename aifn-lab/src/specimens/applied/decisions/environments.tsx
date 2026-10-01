import type { Specimen } from '../../../specimen'
import { MazeAgentSpecimen } from './_environments/maze'

export const specimens: Specimen[] = [
  {
    module: 'applied/decisions',
    title: 'Environment × agent: a maze',
    description:
      'The environment protocol on a maze: a random or Q-learning agent in one rollout, step by step, against value iteration on the maze’s model.',
    tags: ['environment', 'agent', 'rollout', 'Q-learning', 'value iteration', 'maze', 'trace'],
    render: () => <MazeAgentSpecimen />,
  },
]

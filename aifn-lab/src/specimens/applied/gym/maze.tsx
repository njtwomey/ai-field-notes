import type { Specimen } from '../../../specimen'
import { MazeAgentSpecimen } from './_maze/maze'
import { MazeTrainerSpecimen } from './_trainer/trainers'

export const specimens: Specimen[] = [
  {
    module: 'applied/gym',
    title: 'Environment × agent: a maze',
    description:
      'The environment protocol on a maze: a random or Q-learning agent in one rollout, step by step, against value iteration on the maze’s model.',
    tags: ['environment', 'agent', 'rollout', 'Q-learning', 'value iteration', 'maze', 'trace'],
    render: () => <MazeAgentSpecimen />,
  },
  {
    module: 'applied/gym',
    title: 'Gym trainer: a maze',
    description:
      'Headless training in the worker with a streamed learning curve; pick an episode to replay it as trained or evaluate the greedy policy of that moment.',
    tags: ['training', 'replay', 'evaluation', 'Q-learning', 'SARSA', 'REINFORCE', 'maze', 'worker'],
    render: () => <MazeTrainerSpecimen />,
  },
]

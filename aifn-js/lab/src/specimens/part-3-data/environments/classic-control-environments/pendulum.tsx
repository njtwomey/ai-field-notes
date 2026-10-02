import type { Specimen } from '@lab/specimen'
import { PendulumAgentSpecimen } from './_pendulum/pendulum'

export const specimens: Specimen[] = [
  {
    module: 'part-3-data/environments/classic-control-environments',
    title: 'Environment × agent: an inverted pendulum',
    description:
      'Gymnasium’s Pendulum-v1 on core’s RK4 solver: an energy swing-up with an LQR hand-over (gain from autodiff Jacobians of the model), LQR alone, or a random agent, played step by step.',
    tags: ['environment', 'agent', 'rollout', 'pendulum', 'classic control', 'LQR', 'swing-up', 'autodiff', 'trace'],
    render: () => <PendulumAgentSpecimen />,
  },
]

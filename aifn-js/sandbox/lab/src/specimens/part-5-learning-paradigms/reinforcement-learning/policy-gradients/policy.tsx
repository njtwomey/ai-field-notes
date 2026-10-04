import type { Specimen } from '@lab/specimen'
import { DdpgPendulum, OfflineCql, PolicyGradientCartPole } from './_policy/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/reinforcement-learning/policy-gradients',
    title: 'Policy gradients on the gym',
    description:
      'REINFORCE with a baseline, A2C and PPO learning to balance the cart-pole, DDPG learning the pendulum’s continuous swing-up, and offline Q-learning from logged transitions with and without CQL’s conservative term.',
    tags: [
      'policy gradient',
      'REINFORCE',
      'actor–critic',
      'A2C',
      'PPO',
      'DDPG',
      'CQL',
      'offline RL',
      'cart-pole',
      'pendulum',
    ],
    render: () => (
      <>
        <PolicyGradientCartPole />
        <DdpgPendulum />
        <OfflineCql />
      </>
    ),
  },
]

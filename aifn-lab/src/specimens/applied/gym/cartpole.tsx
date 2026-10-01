import type { Specimen } from '../../../specimen'
import { CartPoleTrainerSpecimen } from './_cartpole/cartpole'

export const specimens: Specimen[] = [
  {
    module: 'applied/gym',
    title: 'Environment × agent: a cart-pole',
    description:
      'Gymnasium’s CartPole-v1 trained headlessly in the worker: the cross-entropy method over linear policies, LQR bang-bang or random; pick any episode on the learning curve and play it.',
    tags: ['environment', 'agent', 'cart-pole', 'classic control', 'cross-entropy method', 'LQR', 'training'],
    render: () => <CartPoleTrainerSpecimen />,
  },
]

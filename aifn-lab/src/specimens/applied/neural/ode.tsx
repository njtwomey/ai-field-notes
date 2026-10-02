import type { Specimen } from '../../../specimen'
import { NeuralOdeShowcase } from './_ode/showcase-ode'

export const specimens: Specimen[] = [
  {
    module: 'applied/neural/ode',
    title: 'Showcase: neural ODEs and their variants',
    description:
      'Neural ODE, augmented and second-order NODEs and the ResNet they discretise, trained in the worker on nested circles, moons, spirals and the reflection g(x) = −x; the learned flow over time, backprop through the solver against the adjoint, Euler, RK4 and Dormand–Prince with their work per iteration, RNODE regularisers, a continuous normalising flow (exact and Hutchinson traces) and a latent ODE on irregular trajectories.',
    tags: ['showcase', 'neural ODE', 'ANODE', 'adjoint', 'CNF', 'FFJORD', 'latent ODE', 'ResNet', 'training'],
    render: () => <NeuralOdeShowcase />,
  },
]

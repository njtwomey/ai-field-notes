import type { Specimen } from '../specimen'
import { RosenbrockSpecimen } from './optim/_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim',
    title: 'Optimisers on Rosenbrock',
    description:
      'Twelve optimisers traced from (−1.2, 1) on the Rosenbrock valley: the path over log(1 + f), f against the step, and every internal of the state (gradient, direction, line search, curvature pairs, simplex).',
    tags: ['trace', 'TraceView', 'gradient descent', 'Adam', 'Newton', 'BFGS', 'L-BFGS', 'trust region', 'Nelder–Mead'],
    render: () => <RosenbrockSpecimen />,
  },
]

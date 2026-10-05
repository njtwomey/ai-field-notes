import type { Specimen } from '@lab/specimen'
import { ConvergenceSpecimen, PathCloudSpecimen } from './_sde/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-calculus',
    title: 'Path clouds and their density',
    description: 'Thin sample paths of three SDEs, the histogram of their end points and the Fokker–Planck density.',
    tags: ['Euler–Maruyama', 'Milstein', 'particles', 'Fokker–Planck', 'double well', 'Ornstein–Uhlenbeck'],
    render: () => <PathCloudSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/probability/stochastic-calculus',
    title: 'Strong and weak orders of convergence',
    description:
      'Pathwise and mean errors of Euler–Maruyama, Milstein and stochastic Runge–Kutta on geometric Brownian motion.',
    tags: ['strong order', 'weak order', 'geometric Brownian motion', 'convergence'],
    render: () => <ConvergenceSpecimen />,
  },
]

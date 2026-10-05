import type { Specimen } from '@lab/specimen'
import { ConvergenceComparisonSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/optimisation/second-order-and-quasi-newton',
    title: 'L-BFGS against gradient descent',
    description:
      'f − f* against evaluations for gradient descent, Nesterov, conjugate gradient, L-BFGS, BFGS and Newton on an ill-conditioned bowl or Rosenbrock; change κ and the L-BFGS memory.',
    tags: ['convergence rate', 'condition number', 'L-BFGS', 'conjugate gradient'],
    render: () => <ConvergenceComparisonSpecimen />,
  },
]

import type { Specimen } from '../specimen'
import {
  CmaEsSpecimen,
  ConvergenceComparisonSpecimen,
  LineSearchSpecimen,
  NelderMeadSpecimen,
  ProximalSpecimen,
  RosenbrockSpecimen,
} from './_optim/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim',
    title: 'Optimisers on Rosenbrock',
    description:
      'Twelve optimisers traced from (−1.2, 1) on the Rosenbrock valley: the path over log(1 + f), f against the step, and every internal of the state (gradient, direction, line search, curvature pairs, simplex).',
    tags: ['trace', 'TraceView', 'gradient descent', 'Adam', 'Newton', 'BFGS', 'L-BFGS', 'trust region', 'Nelder–Mead'],
    render: () => <RosenbrockSpecimen />,
  },
  {
    module: 'optim',
    title: 'Line-search trials',
    description:
      'Gradient descent with a backtracking (Armijo) or strong Wolfe line search: the trial points of each step on the surface and on φ(α), with the sufficient-decrease line.',
    tags: ['line search', 'Armijo', 'Wolfe', 'backtracking'],
    render: () => <LineSearchSpecimen />,
  },
  {
    module: 'optim',
    title: 'L-BFGS against gradient descent',
    description:
      'f − f* against evaluations for gradient descent, Nesterov, conjugate gradient, L-BFGS, BFGS and Newton on an ill-conditioned bowl or Rosenbrock; change κ and the L-BFGS memory.',
    tags: ['convergence rate', 'condition number', 'L-BFGS', 'conjugate gradient'],
    render: () => <ConvergenceComparisonSpecimen />,
  },
  {
    module: 'optim',
    title: 'Nelder–Mead simplex',
    description: "The Nelder–Mead simplex on Himmelblau's function, with the operation of each step.",
    tags: ['derivative free', 'simplex'],
    render: () => <NelderMeadSpecimen />,
  },
  {
    module: 'optim',
    title: 'CMA-ES population',
    description: 'CMA-ES on the Rastrigin function: each generation’s samples, the mean path and the step size σ.',
    tags: ['evolution strategy', 'stochastic', 'stream', 'global optimisation'],
    render: () => <CmaEsSpecimen />,
  },
  {
    module: 'optim',
    title: 'ISTA and FISTA',
    description: 'Proximal gradient with soft thresholding on a lasso, with and without Nesterov acceleration.',
    tags: ['proximal', 'lasso', 'L1', 'acceleration'],
    render: () => <ProximalSpecimen />,
  },
]

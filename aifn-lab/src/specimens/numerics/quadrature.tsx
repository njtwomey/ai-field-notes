import type { Specimen } from '../../specimen'
import {
  AdaptiveSpecimen,
  ErrorVsNSpecimen,
  GaussNodesSpecimen,
  InfiniteLimitsSpecimen,
  MonteCarloSpecimen,
} from './_quadrature/figures'

export const specimens: Specimen[] = [
  {
    module: 'numerics/quadrature',
    title: 'Quadrature error against n',
    description:
      'Trapezoid, Simpson, Gauss–Legendre, Romberg and adaptive Gauss–Kronrod on smooth, kinked, singular and oscillating integrands.',
    tags: ['convergence rate', 'trapezoid', 'Simpson', 'Gauss–Legendre', 'Romberg'],
    render: () => <ErrorVsNSpecimen />,
  },
  {
    module: 'numerics/quadrature',
    title: 'Adaptive Gauss–Kronrod',
    description:
      'The subdivision of globally adaptive Gauss–Kronrod 7–15, step by step, with estimated and actual error.',
    tags: ['trace', 'TracePanel', 'adaptive', 'error estimate', 'QUADPACK'],
    render: () => <AdaptiveSpecimen />,
  },
  {
    module: 'numerics/quadrature',
    title: 'Monte Carlo and quasi–Monte Carlo',
    description:
      'Error and standard error against n for Monte Carlo and randomly shifted Sobol and Halton points, in d dimensions.',
    tags: ['Monte Carlo', 'Sobol', 'Halton', 'standard error', 'stream'],
    render: () => <MonteCarloSpecimen />,
  },
  {
    module: 'numerics/quadrature',
    title: 'Gaussian nodes and weights',
    description:
      'Nodes and weights of the Gauss–Legendre, Gauss–Hermite and Gauss–Laguerre rules, with an exactness check.',
    tags: ['Gauss–Legendre', 'Gauss–Hermite', 'Gauss–Laguerre', 'orthogonal polynomials'],
    render: () => <GaussNodesSpecimen />,
  },
  {
    module: 'numerics/quadrature',
    title: 'Infinite limits',
    description:
      'integrate on infinite intervals by the QAGI change of variables, with estimated against actual errors.',
    tags: ['integrate', 'infinite interval', 'error estimate'],
    render: () => <InfiniteLimitsSpecimen />,
  },
]

import type { Specimen } from '../specimen'
import { ConvergenceOrderSpecimen, FixedPointSpecimen, RootStepsSpecimen, WilkinsonSpecimen } from './_solve/figures'

export const specimens: Specimen[] = [
  {
    module: 'solve',
    title: 'Root-finding steps',
    description:
      'Bisection, regula falsi (plain and Illinois), Brent, secant and Newton on four equations: the estimate, bracket and interpolating line at each step.',
    tags: ['trace', 'TraceView', 'bisection', 'Brent', 'secant', 'Newton', 'regula falsi'],
    render: () => <RootStepsSpecimen />,
  },
  {
    module: 'solve',
    title: 'Orders of convergence',
    description: 'The error of every scalar root finder against evaluations of f, on one log-scale chart.',
    tags: ['convergence rate', 'root finding'],
    render: () => <ConvergenceOrderSpecimen />,
  },
  {
    module: 'solve',
    title: "Wilkinson's polynomial",
    description: 'Companion-matrix roots of ∏(x − k) before and after a tiny change to one coefficient.',
    tags: ['polynomial roots', 'companion matrix', 'conditioning', 'eigenvalues'],
    render: () => <WilkinsonSpecimen />,
  },
  {
    module: 'solve',
    title: 'Fixed-point iteration',
    description: 'A cobweb of x ← (1 − ω)x + ωg(x) with the observed contraction factor and error bound at each step.',
    tags: ['fixed point', 'contraction', 'relaxation', 'diagnostics'],
    render: () => <FixedPointSpecimen />,
  },
]

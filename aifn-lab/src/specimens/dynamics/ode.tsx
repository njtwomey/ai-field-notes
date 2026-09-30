import type { Specimen } from '../../specimen'
import { AdaptiveTraceSpecimen, EnergyDriftSpecimen, StabilityRegionSpecimen, StiffStepsSpecimen } from './_ode/figures'

export const specimens: Specimen[] = [
  {
    module: 'dynamics/ode',
    title: 'Step sizes on a stiff problem',
    description:
      'Dormand–Prince against implicit Euler and BDF2 on x′ = −λ(x − cos t): stability, not accuracy, sets the explicit step.',
    tags: ['stiffness', 'Dormand–Prince', 'BDF', 'implicit Euler', 'step size'],
    render: () => <StiffStepsSpecimen />,
  },
  {
    module: 'dynamics/ode',
    title: 'Energy drift on the pendulum',
    description: 'Symplectic Euler, leapfrog and velocity Verlet keep the energy error bounded; RK4 and Euler drift.',
    tags: ['symplectic', 'leapfrog', 'velocity Verlet', 'Hamiltonian', 'energy'],
    render: () => <EnergyDriftSpecimen />,
  },
  {
    module: 'dynamics/ode',
    title: 'Stability regions',
    description: 'The amplification of explicit, implicit and BDF methods on x′ = λx over the complex z = hλ plane.',
    tags: ['stability', 'A-stability', 'Runge–Kutta', 'BDF'],
    render: () => <StabilityRegionSpecimen />,
  },
  {
    module: 'dynamics/ode',
    title: 'Dormand–Prince step by step',
    description: 'An adaptive RK45 run as a trace: accepted steps, step sizes, error estimates and rejected attempts.',
    tags: ['trace', 'TraceView', 'adaptive', 'error control', 'RK45'],
    render: () => <AdaptiveTraceSpecimen />,
  },
]

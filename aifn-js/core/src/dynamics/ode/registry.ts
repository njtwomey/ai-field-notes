/**
 * The algorithms of `aifn/dynamics/ode`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as adaptive from './adaptive'
import * as explicit from './explicit'
import * as implicit from './implicit'
import * as symplectic from './symplectic'
import * as variableBdf from './variable-bdf'

const algorithm = definer<AlgorithmInfo>('algorithm', 'dynamics/ode')

algorithm(
  {
    key: 'rungeKutta',
    name: 'Explicit Runge–Kutta',
    summary: 'A fixed-step explicit Runge–Kutta method from a Butcher tableau (Euler, Heun, midpoint, RK4).',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  explicit.rungeKutta,
)
algorithm(
  {
    key: 'dormandPrince',
    name: 'Dormand–Prince',
    summary: 'The adaptive 5(4) embedded Runge–Kutta pair with error-controlled step sizes.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'error', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['dormand1980'],
  },
  adaptive.dormandPrince,
)
algorithm(
  {
    key: 'implicitEuler',
    name: 'Implicit Euler',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.implicitEuler,
)
algorithm(
  {
    key: 'implicitTrapezoid',
    name: 'Implicit trapezoid',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.implicitTrapezoid,
)
algorithm(
  {
    key: 'bdf',
    name: 'Backward differentiation formula',
    summary: 'The BDF methods of order 1 to 3 for stiff problems.',
    problem: 'ode',
    state: { iterate: 'x', stepSize: 'stepSize', flags: ['converged', 'diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  implicit.bdf,
)
algorithm(
  {
    key: 'adaptiveBdf',
    name: 'Adaptive BDF',
    summary:
      'Variable-order (1–5), variable-step backward differentiation formulae (NDF) for stiff problems, as scipy BDF.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'error', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['shampine1997'],
  },
  variableBdf.adaptiveBdf,
)
algorithm(
  {
    key: 'symplectic',
    name: 'Symplectic integrator',
    summary: 'Symplectic Euler, leapfrog and higher-order splittings for separable Hamiltonians.',
    problem: 'ode',
    state: { iterate: 'x', objective: 'energy', stepSize: 'stepSize', flags: ['diverged'] },
    notes: ['numerical-ode-solvers'],
    cite: ['hairer1993'],
  },
  symplectic.symplectic,
)

/** Every algorithm of the module, keyed by factory name. */
export const odeAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', adaptive, explicit, implicit, symplectic, variableBdf) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

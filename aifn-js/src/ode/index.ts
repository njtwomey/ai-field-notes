/**
 * `aifn/ode`: initial-value problems x′ = f(t, x), every solver a traceable `Algorithm` (see `aifn/trace`) whose state
 * reports the time, the step size, the local error estimate, the evaluations of f and of its Jacobian, and failure.
 *
 * - Explicit Runge–Kutta from Butcher tableaux: `rungeKutta(f, 'euler' | 'heun' | 'midpoint' | 'rk4' | tableau, …)`.
 * - Adaptive: `dormandPrince` (RK45 with error control; each state lists its attempted steps).
 * - Implicit, for stiff systems: `implicitEuler`, `trapezoid`, `bdf` (orders 1–3); Newton via `aifn/solve`, Jacobians
 *   via `aifn/autodiff`.
 * - Symplectic, for separable Hamiltonians: `symplectic(H, 'symplectic-euler' | 'leapfrog' | 'velocity-verlet', …)`
 *   with energy tracking; `hamiltonianSystem` gives the same system to any other solver.
 * - Events: `withEvents(solver, f, events)` locates zero crossings (optionally terminal).
 * - Linear systems: `expm` (Padé, scaling and squaring), `linearFlow`, and `eig` (the general real eigenproblem).
 * - Stability: `amplification`, `stabilityRegion`, `boundaryLocus`.
 * - `solveIvp` runs a solver by name over an interval and returns tensors.
 */

export type { FixedStepOptions, InitialValue, JacobianOption, OdeState, Rhs } from './types'
export { EULER, HEUN, MIDPOINT, RK4, TABLEAUX, rungeKutta, type ButcherTableau } from './explicit'
export { DORMAND_PRINCE, dormandPrince, type AdaptiveOptions, type AdaptiveState, type StepAttempt } from './adaptive'
export { bdf, implicitEuler, trapezoid, type ImplicitOptions, type ImplicitState } from './implicit'
export {
  hamiltonianSystem,
  symplectic,
  type PhaseInitial,
  type SeparableHamiltonian,
  type SymplecticMethod,
  type SymplecticState,
} from './symplectic'
export { withEvents, type EventHit, type EventState, type OdeEvent } from './events'
export { eig, type Eigen } from './eigen'
export { expm, linearFlow, type MatrixExponential } from './linear'
export { amplification, boundaryLocus, stabilityRegion, type StabilityMethod, type StabilityRegion } from './stability'
export { solveIvp, type OdeMethod, type OdeSolution, type SolveIvpOptions } from './solve'

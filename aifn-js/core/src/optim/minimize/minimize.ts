/**
 * `minimize`: run any optimiser in this module by name, as a `run` wrapper over its `Algorithm` (after scipy's
 * `optimize.minimize`; Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17).
 */

import type { Vector } from 'aifn/foundation/tensor'
import { run, type Algorithm } from 'aifn/foundation/trace'
import { conjugateGradient, type ConjugateGradientOptions } from 'aifn/optim/first-order'
import { coordinateDescent, type CoordinateDescentOptions } from 'aifn/optim/first-order'
import {
  adagrad,
  adam,
  adamw,
  gradientDescent,
  momentum,
  nesterov,
  rmsprop,
  type AdamOptions,
  type AdaptiveOptions,
  type GradientDescentOptions,
  type MomentumOptions,
  type RmspropOptions,
} from 'aifn/optim/first-order'
import { objectiveFn, valueFunction, type RunOptions, type StartOptions } from '../options'
import { nelderMead, type NelderMeadOptions } from 'aifn/optim/derivative-free'
import { newton, trustRegion, type NewtonOptions, type TrustRegionOptions } from 'aifn/optim/second-order'
import { bfgs, lbfgs, type LbfgsOptions, type QuasiNewtonOptions } from 'aifn/optim/second-order'
import {
  cmaEs,
  simulatedAnnealing,
  type CmaEsOptions,
  type SimulatedAnnealingOptions,
} from 'aifn/optim/derivative-free'
import type {
  IterateState,
  Objective,
  ObjectiveFn,
  Scalar,
  Size,
  ValueFunction,
  VectorLike,
} from 'aifn/foundation/contracts'
import type { Tensor } from 'aifn/foundation/tensor'

/** Options of each method `minimize` can run, by name. */
export type MethodOptions = {
  'gradient-descent': GradientDescentOptions
  momentum: MomentumOptions
  nesterov: MomentumOptions
  adagrad: AdaptiveOptions
  rmsprop: RmspropOptions
  adam: AdamOptions
  adamw: AdamOptions
  newton: NewtonOptions
  'trust-region': TrustRegionOptions
  bfgs: QuasiNewtonOptions
  lbfgs: LbfgsOptions
  'conjugate-gradient': ConjugateGradientOptions
  'coordinate-descent': CoordinateDescentOptions
  'nelder-mead': NelderMeadOptions
  'simulated-annealing': SimulatedAnnealingOptions
  'cma-es': CmaEsOptions
}

/** A method name for `minimize`. */
export type Method = keyof MethodOptions

/** The result of `minimize`. */
export type MinimizeResult<S extends IterateState = IterateState> = {
  method: Method
  /** The final iterate (for simulated annealing, the best point seen). */
  x: Vector
  value: Scalar
  converged: boolean
  diverged: boolean
  steps: Size
  evaluations: Size
  /** The final state of the algorithm, with all its internals. */
  state: S
}

/**
 * Minimises f from `x0` with the named method (default `'lbfgs'`) for at most `maxSteps` steps (default 1000),
 * returning the final iterate and state. `f` is an `Objective` (its value written with primitives; gradients by
 * autodiff) or a bare function: `{ value, grad }` for gradient methods, or a plain number for `'nelder-mead'`,
 * `'simulated-annealing'` and `'cma-es'`. `'newton'` and `'trust-region'` need a `hessian` option. Stochastic methods
 * draw from `stream`.
 */
export function minimize<M extends Method = 'lbfgs'>(
  f: Objective<Tensor> | ObjectiveFn | ValueFunction,
  x0: VectorLike,
  options: { method?: M } & RunOptions & MethodOptions[M] = {} as MethodOptions[M],
): MinimizeResult {
  const method = (options.method ?? 'lbfgs') as Method
  const { maxSteps = 1000, stream } = options
  const gradient = objectiveFn(f as Objective<Tensor> | ObjectiveFn)
  const value = valueFunction(f)
  const o = options as never
  const algorithms: Record<Method, () => Algorithm<StartOptions, IterateState>> = {
    'gradient-descent': () => gradientDescent(gradient, o),
    momentum: () => momentum(gradient, o),
    nesterov: () => nesterov(gradient, o),
    adagrad: () => adagrad(gradient, o),
    rmsprop: () => rmsprop(gradient, o),
    adam: () => adam(gradient, o),
    adamw: () => adamw(gradient, o),
    newton: () => newton(gradient, o),
    'trust-region': () => trustRegion(gradient, o),
    bfgs: () => bfgs(gradient, o),
    lbfgs: () => lbfgs(gradient, o),
    'conjugate-gradient': () => conjugateGradient(gradient, o),
    'coordinate-descent': () => coordinateDescent(gradient, o),
    'nelder-mead': () => nelderMead(value, o),
    'simulated-annealing': () => simulatedAnnealing(value, o),
    'cma-es': () => cmaEs(value, o),
  }
  if (!(method in algorithms)) throw new Error(`minimize: unknown method "${method}"`)
  const state = run(algorithms[method](), { x0 }, maxSteps, { stream })
  const annealing = method === 'simulated-annealing'
  const s = state as IterateState & { best?: Vector; bestValue?: number }
  return {
    method,
    x: annealing && s.best ? s.best : state.x,
    value: annealing && s.bestValue !== undefined ? s.bestValue : state.value,
    converged: state.converged,
    diverged: state.diverged,
    steps: state.t,
    evaluations: state.evaluations,
    state,
  }
}

/**
 * First-order methods: gradient descent (fixed, scheduled or line-searched step), heavy-ball momentum, Nesterov's
 * accelerated gradient, AdaGrad, RMSProp and Adam / AdamW. They share one algorithm skeleton and differ in the update
 * rule, which reads the gradient and the method's slots (velocity, moment estimates) and returns the step.
 */

import type { Algorithm } from 'aifn/trace'
import type { Vector } from 'aifn/tensor'
import {
  backtrackingSearch,
  strongWolfeSearch,
  type BacktrackingOptions,
  type LineSearchResult,
  type StrongWolfeOptions,
} from './lineSearch'
import type { IterateState, Objective, Schedule, StoppingOptions, VectorLike } from './types'
import {
  DEFAULT_DIVERGE,
  DEFAULT_TOLERANCE,
  axpy,
  data,
  divergedAt,
  evaluate,
  norm,
  sub,
  toF64,
  vec,
  type F64,
} from './vector'

/** The state of a first-order method. */
export type FirstOrderState = IterateState & {
  /** ∇f(x). */
  grad: Vector
  gradNorm: number
  /** The step applied last, x_t − x_{t−1} (zeros at t = 0). */
  update: Vector
  /** The step size (learning rate, or accepted line-search step) used on the last step; NaN at t = 0. */
  stepSize: number
  /**
   * The method's running quantities: `velocity` (momentum, Nesterov), `lookahead` and `lookaheadGrad` (Nesterov: the
   * point x + μv where the gradient was taken, and that gradient), `sumSquares` (AdaGrad), `meanSquare` (RMSProp),
   * `firstMoment` and `secondMoment` (Adam, before bias correction). Empty for gradient descent.
   */
  slots: Record<string, Vector>
  /** The last line search (gradient descent with `lineSearch` only), with its trial points; null otherwise. */
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. Line-searched descent only. */
  stalled: boolean
}

/** Options every first-order method takes. */
export type FirstOrderOptions = StoppingOptions & {
  /** Step size η, or a schedule t ↦ η_t (see `inverseTimeDecay`, `exponentialDecay`, `inverseSqrtDecay`). */
  lr?: number | Schedule
}

/** The starting point. */
export type StartOptions = { x0: VectorLike }

type RuleInput = { f: Objective; x: F64; grad: F64; slots: Record<string, F64>; lr: number; t: number }
type RuleOutput = { update: F64; slots: Record<string, F64>; evaluations?: number }
type Rule = (input: RuleInput) => RuleOutput

const rate = (lr: number | Schedule, t: number) => (typeof lr === 'function' ? lr(t) : lr)

/** The shared skeleton: evaluate at x0, then x ← x + update(rule) and re-evaluate. */
function firstOrder(
  name: string,
  f: Objective,
  options: FirstOrderOptions,
  initSlots: (n: number) => Record<string, F64>,
  rule: Rule,
  defaultLr: number,
): Algorithm<StartOptions, FirstOrderState> {
  const { lr = defaultLr, tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const wrapSlots = (slots: Record<string, F64>) =>
    Object.fromEntries(Object.entries(slots).map(([k, v]) => [k, vec(v)])) as Record<string, Vector>
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        update: vec(new Float64Array(x.length)),
        stepSize: NaN,
        slots: wrapSlots(initSlots(x.length)),
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s) => {
      const x = data(s.x)
      const slots = Object.fromEntries(Object.entries(s.slots).map(([k, v]) => [k, data(v)]))
      const eta = rate(lr, s.t)
      const out = rule({ f, x, grad: data(s.grad), slots, lr: eta, t: s.t })
      const next = axpy(1, out.update, x)
      const { value, grad } = evaluate(f, next, name)
      const gradNorm = norm(grad)
      return {
        t: s.t + 1,
        x: vec(next),
        value,
        grad: vec(grad),
        gradNorm,
        update: vec(out.update),
        stepSize: eta,
        slots: wrapSlots(out.slots),
        lineSearch: null,
        stalled: false,
        evaluations: s.evaluations + 1 + (out.evaluations ?? 0),
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, next, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

const neg = (lr: number, g: F64) => {
  const out = new Float64Array(g.length)
  for (let i = 0; i < g.length; i++) out[i] = -lr * g[i]
  return out
}

/** Options for `gradientDescent`. */
export type GradientDescentOptions = FirstOrderOptions & {
  /**
   * Choose each step by a line search along −∇f starting from α₀ = `lr` (at step t): backtracking to the Armijo
   * condition, or a strong Wolfe search. Off by default (fixed or scheduled step).
   */
  lineSearch?: 'backtracking' | 'strong-wolfe'
  /** Options for the line search. */
  lineSearchOptions?: BacktrackingOptions & StrongWolfeOptions
}

/**
 * Gradient descent, x ← x − η_t∇f(x) (Cauchy, 1847), with a fixed step, a schedule, or a line search along the
 * negative gradient. `f` returns `{ value, grad }`; `init` takes `{ x0 }`. Default η = 0.01.
 *
 * On a quadratic with Hessian eigenvalues in [μ, L], a fixed step η < 2/L converges linearly, with rate
 * max(|1 − ημ|, |1 − ηL|) per step.
 */
export function gradientDescent(
  f: Objective,
  options: GradientDescentOptions = {},
): Algorithm<StartOptions, FirstOrderState> {
  const { lineSearch } = options
  if (!lineSearch)
    return firstOrder(
      'gradient-descent',
      f,
      options,
      () => ({}),
      ({ grad, lr }) => ({ update: neg(lr, grad), slots: {} }),
      0.01,
    )

  const { lr = 1, tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const base = firstOrder(
    'gradient-descent',
    f,
    options,
    () => ({}),
    () => ({ update: new Float64Array(0), slots: {} }),
    1,
  )
  return {
    ...base,
    step: (s) => {
      const x = data(s.x)
      const g = data(s.grad)
      const p = neg(1, g)
      const alpha0 = rate(lr, s.t)
      const search =
        lineSearch === 'backtracking'
          ? backtrackingSearch(f, x, s.value, g, p, { ...options.lineSearchOptions, alpha0 })
          : strongWolfeSearch(f, x, s.value, g, p, { ...options.lineSearchOptions, alpha0 })
      const gradNorm = norm(search.grad)
      return {
        ...s,
        t: s.t + 1,
        x: vec(search.x),
        value: search.value,
        grad: vec(search.grad),
        gradNorm,
        update: vec(sub(search.x, x)),
        stepSize: search.result.alpha,
        lineSearch: search.result,
        stalled: search.result.alpha === 0,
        evaluations: s.evaluations + search.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(search.value, search.x, divergeAbove),
      }
    },
  }
}

/** Options for `momentum` and `nesterov`. */
export type MomentumOptions = FirstOrderOptions & {
  /** Momentum coefficient μ in [0, 1). Default 0.9. */
  momentum?: number
}

/**
 * Heavy-ball momentum (Polyak, 1964): v ← μv − η∇f(x), x ← x + v. With μ = 0 it is gradient descent. Default η = 0.01,
 * μ = 0.9. The velocity is `slots.velocity`.
 */
export function momentum(f: Objective, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const mu = options.momentum ?? 0.9
  return firstOrder(
    'momentum',
    f,
    options,
    (n) => ({ velocity: new Float64Array(n) }),
    ({ grad, slots, lr }) => {
      const v = slots.velocity
      const velocity = new Float64Array(v.length)
      for (let i = 0; i < v.length; i++) velocity[i] = mu * v[i] - lr * grad[i]
      return { update: velocity, slots: { velocity } }
    },
    0.01,
  )
}

/**
 * Nesterov's accelerated gradient in the form of Sutskever et al. (2013, eq. 3–4): the gradient is taken at the
 * look-ahead point y = x + μv, then v ← μv − η∇f(y) and x ← x + v. The look-ahead point and its gradient are kept in
 * `slots.lookahead` and `slots.lookaheadGrad`; `grad` is always ∇f(x), so each step evaluates f twice.
 */
export function nesterov(f: Objective, options: MomentumOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const mu = options.momentum ?? 0.9
  return firstOrder(
    'nesterov',
    f,
    options,
    (n) => ({ velocity: new Float64Array(n), lookahead: new Float64Array(n), lookaheadGrad: new Float64Array(n) }),
    ({ f: fn, x, slots, lr }) => {
      const v = slots.velocity
      const lookahead = axpy(mu, v, x)
      const { grad: g } = evaluate(fn, lookahead, 'nesterov')
      const velocity = new Float64Array(v.length)
      for (let i = 0; i < v.length; i++) velocity[i] = mu * v[i] - lr * g[i]
      return { update: velocity, slots: { velocity, lookahead, lookaheadGrad: g }, evaluations: 1 }
    },
    0.01,
  )
}

/** Options for the adaptive methods. */
export type AdaptiveOptions = FirstOrderOptions & {
  /** Added to the root in the denominator to avoid division by zero. Default 1e-8. */
  epsilon?: number
}

/**
 * AdaGrad (Duchi, Hazan & Singer, 2011): G ← G + g², x ← x − ηg / (√G + ε), elementwise. The accumulated squares are
 * `slots.sumSquares`. Default η = 0.1.
 */
export function adagrad(f: Objective, options: AdaptiveOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const eps = options.epsilon ?? 1e-8
  return firstOrder(
    'adagrad',
    f,
    options,
    (n) => ({ sumSquares: new Float64Array(n) }),
    ({ grad, slots, lr }) => {
      const sumSquares = new Float64Array(grad.length)
      const update = new Float64Array(grad.length)
      for (let i = 0; i < grad.length; i++) {
        sumSquares[i] = slots.sumSquares[i] + grad[i] * grad[i]
        update[i] = (-lr * grad[i]) / (Math.sqrt(sumSquares[i]) + eps)
      }
      return { update, slots: { sumSquares } }
    },
    0.1,
  )
}

/** Options for `rmsprop`. */
export type RmspropOptions = AdaptiveOptions & {
  /** Decay ρ of the mean of squared gradients. Default 0.9. */
  decay?: number
}

/**
 * RMSProp (Tieleman & Hinton, 2012, Coursera lecture 6.5): s ← ρs + (1 − ρ)g², x ← x − ηg / (√s + ε). The mean square
 * is `slots.meanSquare`. Default η = 0.01, ρ = 0.9.
 */
export function rmsprop(f: Objective, options: RmspropOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const eps = options.epsilon ?? 1e-8
  const rho = options.decay ?? 0.9
  return firstOrder(
    'rmsprop',
    f,
    options,
    (n) => ({ meanSquare: new Float64Array(n) }),
    ({ grad, slots, lr }) => {
      const meanSquare = new Float64Array(grad.length)
      const update = new Float64Array(grad.length)
      for (let i = 0; i < grad.length; i++) {
        meanSquare[i] = rho * slots.meanSquare[i] + (1 - rho) * grad[i] * grad[i]
        update[i] = (-lr * grad[i]) / (Math.sqrt(meanSquare[i]) + eps)
      }
      return { update, slots: { meanSquare } }
    },
    0.01,
  )
}

/** Options for `adam` and `adamw`. */
export type AdamOptions = AdaptiveOptions & {
  /** Decay of the first-moment estimate. Default 0.9. */
  beta1?: number
  /** Decay of the second-moment estimate. Default 0.999. */
  beta2?: number
  /** Weight decay λ. Default 0 for `adam`, 0.01 for `adamw`. */
  weightDecay?: number
  /**
   * Decoupled weight decay (AdamW): x ← x − η(m̂/(√v̂ + ε) + λx). When false, λx is added to the gradient before the
   * moment updates (L2 regularisation as Adam implements it). Default false for `adam`, true for `adamw`.
   */
  decoupled?: boolean
}

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², m̂ = m/(1 − β₁ᵗ),
 * v̂ = v/(1 − β₂ᵗ), x ← x − ηm̂/(√v̂ + ε), with t counting from 1. The raw moments are `slots.firstMoment` and
 * `slots.secondMoment`. Default η = 0.001. With `weightDecay` and `decoupled` it is AdamW (Loshchilov & Hutter, 2019).
 */
export function adam(f: Objective, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  const { epsilon: eps = 1e-8, beta1 = 0.9, beta2 = 0.999, weightDecay = 0, decoupled = false } = options
  return firstOrder(
    decoupled ? 'adamw' : 'adam',
    f,
    options,
    (n) => ({ firstMoment: new Float64Array(n), secondMoment: new Float64Array(n) }),
    ({ x, grad, slots, lr, t }) => {
      const n = grad.length
      const firstMoment = new Float64Array(n)
      const secondMoment = new Float64Array(n)
      const update = new Float64Array(n)
      const c1 = 1 - beta1 ** (t + 1)
      const c2 = 1 - beta2 ** (t + 1)
      for (let i = 0; i < n; i++) {
        const g = decoupled ? grad[i] : grad[i] + weightDecay * x[i]
        firstMoment[i] = beta1 * slots.firstMoment[i] + (1 - beta1) * g
        secondMoment[i] = beta2 * slots.secondMoment[i] + (1 - beta2) * g * g
        const mHat = firstMoment[i] / c1
        const vHat = secondMoment[i] / c2
        update[i] = -lr * (mHat / (Math.sqrt(vHat) + eps) + (decoupled ? weightDecay * x[i] : 0))
      }
      return { update, slots: { firstMoment, secondMoment } }
    },
    0.001,
  )
}

/** AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay, default λ = 0.01. See `adam`. */
export function adamw(f: Objective, options: AdamOptions = {}): Algorithm<StartOptions, FirstOrderState> {
  return adam(f, { weightDecay: 0.01, ...options, decoupled: options.decoupled ?? true })
}

/** The schedule η_t = η₀ / (1 + kt). */
export const inverseTimeDecay =
  (lr: number, k: number): Schedule =>
  (t) =>
    lr / (1 + k * t)

/** The schedule η_t = η₀γᵗ. */
export const exponentialDecay =
  (lr: number, gamma: number): Schedule =>
  (t) =>
    lr * gamma ** t

/** The schedule η_t = η₀ / √(t + 1). */
export const inverseSqrtDecay =
  (lr: number): Schedule =>
  (t) =>
    lr / Math.sqrt(t + 1)

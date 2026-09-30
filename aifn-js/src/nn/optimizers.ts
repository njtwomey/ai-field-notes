/**
 * Optimisers over parameter pytrees: SGD (with momentum, Nesterov momentum and weight decay) and Adam / AdamW. Each is
 * a pure pair `init(params)` → slots and `update(params, grads, slots, t)` → new params and slots, applied leaf by leaf,
 * so a training step is a pure function of its state.
 *
 * `aifn/optim` has the same methods as traceable algorithms over one flat vector with a deterministic objective. A
 * training loop needs the update rule alone (the objective changes with the minibatch at every step), so the rules
 * are written here for trees; see the module docs for what `optim` would need to serve both.
 */

import { add, div, mul, sqrt, square, sub, type Tensor } from 'aifn/tensor'
import { treeMap, treeZip, type LeafValue, type Params } from './tree'

/** A learning rate, or a schedule t ↦ η_t for the step t = 0, 1, 2, … */
export type LearningRate = number | ((t: number) => number)

const rateAt = (lr: LearningRate, t: number) => (typeof lr === 'function' ? lr(t) : lr)

/** An optimiser's running quantities, one tree per slot (e.g. `velocity`, or `firstMoment` and `secondMoment`). */
export type Slots = Record<string, Params>

/** An optimiser over parameter trees. */
export interface Optimizer {
  readonly name: string
  /** The slots at the start (zeros shaped like the parameters). */
  init(params: Params): Slots
  /** One step from gradients `grads` (same structure as `params`) at step t; pure. */
  update(params: Params, grads: Params, slots: Slots, t: number): { params: Params; slots: Slots }
}

const zerosLike = (params: Params): Params => treeMap(params, (x) => (typeof x === 'number' ? 0 : mul(x, 0)))

/** Leaf arithmetic that keeps numbers as numbers and tensors as tensors. */
const leaf = (v: unknown) => v as LeafValue & Tensor

/** Options of `sgd`. */
export type SgdOptions = {
  lr?: LearningRate
  /** Momentum μ (default 0: plain SGD). */
  momentum?: number
  /** Use Nesterov momentum (Sutskever et al., 2013). */
  nesterov?: boolean
  /** L2 penalty λ added to the gradient, λθ (default 0). */
  weightDecay?: number
}

/**
 * Stochastic gradient descent with momentum, in PyTorch's form: g ← ∇ + λθ, v ← μv + g, θ ← θ − η·(g + μv if Nesterov,
 * else v).
 */
export function sgd({ lr = 0.1, momentum = 0, nesterov = false, weightDecay = 0 }: SgdOptions = {}): Optimizer {
  return {
    name: momentum > 0 ? (nesterov ? 'sgd-nesterov' : 'sgd-momentum') : 'sgd',
    init: (params): Slots => (momentum > 0 ? { velocity: zerosLike(params) } : {}),
    update: (params, grads, slots, t) => {
      const eta = rateAt(lr, t)
      const g =
        weightDecay === 0 ? grads : treeZip([grads, params], ([gi, p]) => add(leaf(gi), mul(weightDecay, leaf(p))))
      if (momentum === 0) {
        return { params: treeZip([params, g], ([p, gi]) => sub(leaf(p), mul(eta, leaf(gi)))), slots }
      }
      const velocity = treeZip([slots.velocity, g], ([v, gi]) => add(mul(momentum, leaf(v)), leaf(gi)))
      const step = nesterov ? treeZip([g, velocity], ([gi, v]) => add(leaf(gi), mul(momentum, leaf(v)))) : velocity
      return { params: treeZip([params, step], ([p, s]) => sub(leaf(p), mul(eta, leaf(s)))), slots: { velocity } }
    },
  }
}

/** Options of `adam`. */
export type AdamOptions = {
  lr?: LearningRate
  beta1?: number
  beta2?: number
  epsilon?: number
  /** Weight decay λ: added to the gradient (Adam) or applied to θ directly (`decoupled`, AdamW). Default 0. */
  weightDecay?: number
  /** Decoupled weight decay (AdamW; Loshchilov & Hutter, 2019). */
  decoupled?: boolean
}

/**
 * Adam (Kingma & Ba, 2015, Algorithm 1): m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², θ ← θ − η·m̂/(√v̂ + ε) with the
 * bias-corrected m̂ = m/(1 − β₁^{t+1}), v̂ = v/(1 − β₂^{t+1}). With `decoupled`, θ also shrinks by ηλθ (AdamW).
 */
export function adam(options: AdamOptions = {}): Optimizer {
  const { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0, decoupled = false } = options
  return {
    name: decoupled ? 'adamw' : 'adam',
    init: (params) => ({ firstMoment: zerosLike(params), secondMoment: zerosLike(params) }),
    update: (params, grads, slots, t) => {
      const eta = rateAt(lr, t)
      const g =
        weightDecay === 0 || decoupled
          ? grads
          : treeZip([grads, params], ([gi, p]) => add(leaf(gi), mul(weightDecay, leaf(p))))
      const m = treeZip([slots.firstMoment, g], ([mi, gi]) => add(mul(beta1, leaf(mi)), mul(1 - beta1, leaf(gi))))
      const v = treeZip([slots.secondMoment, g], ([vi, gi]) =>
        add(mul(beta2, leaf(vi)), mul(1 - beta2, square(leaf(gi)))),
      )
      const c1 = 1 - beta1 ** (t + 1)
      const c2 = 1 - beta2 ** (t + 1)
      const next = treeZip([params, m, v], ([p, mi, vi]) => {
        const step = div(div(leaf(mi), c1), add(sqrt(div(leaf(vi), c2)), epsilon))
        const decayed = decoupled && weightDecay !== 0 ? sub(leaf(p), mul(eta * weightDecay, leaf(p))) : leaf(p)
        return sub(decayed, mul(eta, step))
      })
      return { params: next, slots: { firstMoment: m, secondMoment: v } }
    },
  }
}

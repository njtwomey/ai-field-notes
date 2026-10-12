/**
 * The penalised cross-entropy of two-feature logistic regression with the bias fixed at 0, and the three solvers of
 * the solver race, on the engine: the loss is `binaryCrossEntropyWithLogits` written with tensor primitives, its
 * gradient and Hessian come from autodiff, and the solvers are `gradientDescent`, `newton` and `sgdRule`.
 */
import { grad, hessian } from 'aifn-compute/foundation/autodiff'
import { permutation, stream } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  matmul,
  mul,
  sum,
  take,
  type Tensor,
  tensor,
  toFlat,
  toRows,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { accuracy } from 'aifn-compute/learning/metrics'
import { conditionNumber, svd } from 'aifn-compute/numerics/linalg'
import { applyUpdates, gradientDescent, newton, objectiveFn, type ObjectiveFn, sgdRule } from 'aifn-compute/optim'
import type { Points } from './datasets'

export type Vec = [number, number]

/** A dataset as tensors (X is n × 2, y holds n labels in {0, 1}) with the L2 penalty λ. */
export type Problem = { X: Tensor; y: Tensor; n: number; lambda: number }

export function problemOf(data: Points, lambda: number): Problem {
  return { X: tensor(data.x.map((v, i) => [v, data.y[i]])), y: tensor(data.group), n: data.group.length, lambda }
}

/** Mean cross-entropy of the rows `X`, `y` at w, plus (λ/2)‖w‖². Written with primitives, so autodiff applies. */
function lossOn(X: Value, y: Value, lambda: number, w: Value): Value {
  const data = binaryCrossEntropyWithLogits(matmul(X, w), y as Tensor)
  return lambda === 0 ? data : add(data, mul(0.5 * lambda, sum(mul(w, w))))
}

/** The penalised loss of the whole dataset at w. */
export const loss = (p: Problem, w: Vec): number => lossOn(p.X, p.y, p.lambda, tensor(w)) as number

/** The loss and its gradient as an `ObjectiveFn`, from autodiff. */
export const objective = (p: Problem): ObjectiveFn =>
  objectiveFn({ kind: 'objective', name: 'logistic loss', dim: 2, value: (w) => lossOn(p.X, p.y, p.lambda, w) })

/** ∇L(w), from autodiff. */
export const gradient = (p: Problem, w: Vec): Vec => {
  const [a, b] = toFlat(grad((v: Tensor) => lossOn(p.X, p.y, p.lambda, v))(tensor(w)) as Tensor)
  return [a, b]
}

/** ∇²L(w) as rows, from autodiff. */
export const hessianAt = (p: Problem, w: Vec): number[][] =>
  toRows(hessian((v: Tensor) => lossOn(p.X, p.y, p.lambda, v))(tensor(w)) as Tensor)

/** The share of examples on the correct side of the boundary w·x = 0 (predicted class 1 when the logit is ≥ 0). */
export const accuracyAt = (p: Problem, w: Vec): number =>
  accuracy(
    toFlat(p.y),
    toFlat(matmul(p.X, tensor(w)) as Tensor).map((z) => (z >= 0 ? 1 : 0)),
  )

/** Condition number of the Hessian at w. */
export const hessianCondition = (p: Problem, w: Vec): number => conditionNumber(tensor(hessianAt(p, w)))

/** The smoothness constant L = λmax(XᵀX)/(4n) + λ, a bound on the curvature since p(1 − p) ≤ 1/4. */
export const smoothness = (p: Problem): number => toFlat(svd(p.X).S)[0] ** 2 / (4 * p.n) + p.lambda

/** Iterates beyond this norm count as diverged, so a runaway path does not stretch the plots. */
const LIMIT = 1e3
const finite = ([a, b]: Vec) => Number.isFinite(a) && Number.isFinite(b) && Math.hypot(a, b) < LIMIT

/** The recorded iterates of a run, up to the first that diverged, padded with the last to `steps` + 1 points
 * when the run stopped early because it converged or stalled. */
function iterates(run: ReturnType<typeof trace>, steps: number): Vec[] {
  const xs = dense.data(run.series.x as Tensor)
  const path: Vec[] = []
  for (let i = 0; i < run.index.length; i++) {
    const w: Vec = [xs[2 * i], xs[2 * i + 1]]
    if (!finite(w)) return path
    path.push(w)
  }
  if (run.meta.stopped === 'diverged') return path
  while (path.length < steps + 1) path.push(path[path.length - 1])
  return path
}

const RECORD = { record: { x: (s: { x: Tensor }) => s.x }, timing: false } as const

/** `steps` steps of gradient descent, w ← w − η∇L(w), from `start`. The gradient-norm test is off. */
export function descend(p: Problem, start: Vec, eta: number, steps: number): Vec[] {
  const alg = gradientDescent(objective(p), { stepSize: eta, tolerance: 0 })
  return iterates(trace(alg, { x0: start }, steps, RECORD), steps)
}

/** `steps` Newton steps −H⁻¹∇L; with `damped`, each is halved until the loss decreases enough (Armijo). */
export function newtonPath(p: Problem, start: Vec, damped: boolean, steps: number): Vec[] {
  const alg = newton(objective(p), {
    hessian: (w) => hessianAt(p, [w.data[0], w.data[1]]),
    lineSearch: damped ? 'backtracking' : 'none',
    tolerance: 0,
  })
  return iterates(trace(alg, { x0: start }, steps, RECORD), steps)
}

/**
 * Mini-batch SGD for `passes` passes over the data, each in a fresh random order from the stream `seed`, with
 * batches of `batch` examples and step η, or η/(1 + pass) with `decay`. Returns every step and the index in `steps`
 * where each pass ends.
 */
export function sgd(p: Problem, start: Vec, eta: number, batch: number, decay: boolean, seed: number, passes: number) {
  const random = stream(seed)
  const perPass = Math.ceil(p.n / batch)
  const rule = sgdRule({ stepSize: (t) => (decay ? eta / (1 + Math.floor(t / perPass)) : eta) })
  let w = tensor(start)
  let state = rule.init(w)
  const steps: Vec[] = [start]
  const ends = [0]
  for (let pass = 0; pass < passes; pass++) {
    const order = toFlat(permutation(random, p.n))
    for (let s = 0; s < p.n; s += batch) {
      const rows = order.slice(s, s + batch)
      const Xb = take(p.X, rows)
      const yb = take(p.y, rows)
      const g = grad((v: Tensor) => lossOn(Xb, yb, p.lambda, v))(w) as Tensor
      const out = rule.update(g, state, w)
      state = out.state
      w = applyUpdates(w, out.updates)
      const next = toFlat(w) as Vec
      if (!finite(next)) return { steps, ends }
      steps.push(next)
    }
    ends.push(steps.length - 1)
  }
  return { steps, ends }
}

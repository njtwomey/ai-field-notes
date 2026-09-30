/**
 * The function transforms: `grad`, `valueAndGrad`, `vjp`, `jvp`, `jacobian`, `hessian`, `hvp` and `stopGradient`.
 * They follow the interface of JAX (Bradbury et al., 2018): a transform takes a function and returns a function.
 */

import {
  add,
  currentTape,
  defineOp,
  isTraced,
  mul,
  ones,
  reshape,
  shapeOfValue,
  stack,
  sum,
  unwrap,
  withTape,
  zeros,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn/tensor'
import { GraphTape, sweep } from './tape'
import { flattenTree, zerosLike, type Flat } from './tree'

const count = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

// ── Sessions: one tape per outermost transform ───────────────────────────────────────────────────────────────────────

/**
 * The tape a transform records on and whether it is nested. Inside another transform (its tape is active) the
 * transform shares that tape, and its reverse sweeps record themselves so the outer transform can differentiate them.
 */
type Session = { tape: GraphTape; nested: boolean }

function openSession(leaves: readonly Value[]): Session {
  const active = currentTape()
  if (active instanceof GraphTape) return { tape: active, nested: true }
  for (const x of leaves) if (isTraced(x) && x.tape instanceof GraphTape) return { tape: x.tape, nested: true }
  return { tape: new GraphTape(), nested: false }
}

/** The raw value of a result at the outermost level; traced values stay traced inside another transform. */
function settle<T>(session: Session, x: T): T {
  if (session.nested) return x
  return (isTraced(x) ? unwrap(x) : x) as T
}

/** Seed cotangent for a scalar output: 1, or a rank-0 tensor of ones. */
function scalarSeed(where: string, y: Value): Value {
  const raw = unwrap(y)
  if (typeof raw === 'number') return 1
  if (raw.shape.length === 0) return ones([])
  throw new Error(`${where}: the function must return a number or a rank-0 tensor, not shape [${raw.shape.join(', ')}]`)
}

/** Replace missing cotangents by zeros of their leaf's shape. */
function filled(cotangents: readonly (Value | null)[], leaves: readonly Value[]): Value[] {
  return cotangents.map((g, i) => g ?? zerosLike(unwrap(leaves[i])))
}

type Argnums = number | readonly number[]

function argnumList(argnums: Argnums, n: number, where: string): number[] {
  const list = typeof argnums === 'number' ? [argnums] : [...argnums]
  for (const k of list) {
    if (!Number.isInteger(k) || k < 0 || k >= n) throw new Error(`${where}: argnum ${k} is out of range for ${n} args`)
  }
  return list
}

/** Record the arguments `argnums` as inputs on the session's tape; returns the new arguments and their flat trees. */
function traceArgs(session: Session, args: readonly unknown[], argnums: readonly number[]) {
  const out = [...args]
  const flats: Flat[] = []
  const inputs: Traced[] = []
  for (const k of argnums) {
    const flat = flattenTree(args[k], `arg${k}`)
    const traced = flat.leaves.map((leaf, i) => session.tape.input(leaf, flat.paths[i]))
    out[k] = flat.rebuild(traced)
    flats.push(flat)
    inputs.push(...traced)
  }
  return { args: out, flats, inputs }
}

/** Split a flat list of cotangents back into one tree per differentiated argument. */
function unflatten(flats: readonly Flat[], cotangents: readonly Value[]): unknown[] {
  let at = 0
  return flats.map((flat) => {
    const part = cotangents.slice(at, at + flat.leaves.length)
    at += flat.leaves.length
    return flat.rebuild(part)
  })
}

// ── grad and valueAndGrad ────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `grad` and `valueAndGrad`. */
export type GradOptions = {
  /** Which argument(s) to differentiate with respect to (default 0). An array gives an array of gradients. */
  argnums?: Argnums
}

/** The result of `valueAndGrad`: the function's value and its gradient. */
export type ValueAndGrad<V, G> = { value: V; grad: G }

/**
 * `valueAndGrad(f)` is a function computing f(...args) and the gradient of f with respect to argument `argnums`
 * (default 0) in one forward and one reverse pass. f must return a number or a rank-0 tensor. Arguments may be
 * numbers, tensors or pytrees (nested arrays and objects) of them; each gradient has its argument's structure, with
 * numbers for numbers and tensors of the same shape for tensors. With `argnums` an array, `grad` is an array of
 * gradients. Inside another transform, values and gradients are traced so that the outer transform differentiates
 * them (nested `grad` gives second derivatives).
 *
 * @example valueAndGrad((w: Value) => sum(square(w)))(tensor([1, 2])) // { value: 5, grad: [2, 4] }
 */
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
): (...args: A) => ValueAndGrad<number | Tensor, A[0]>
export function valueAndGrad<A extends unknown[], N extends number>(
  f: (...args: A) => Value,
  options: { argnums: N },
): (...args: A) => ValueAndGrad<number | Tensor, A[N]>
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
  options: { argnums: readonly number[] },
): (...args: A) => ValueAndGrad<number | Tensor, A[number][]>
export function valueAndGrad<A extends unknown[]>(
  f: (...args: A) => Value,
  options?: GradOptions,
): (...args: A) => ValueAndGrad<number | Tensor, unknown>
export function valueAndGrad<A extends unknown[]>(f: (...args: A) => Value, { argnums = 0 }: GradOptions = {}) {
  return (...args: A) => {
    const nums = argnumList(argnums, args.length, 'grad')
    const session = openSession(nums.flatMap((k) => flattenTree(args[k]).leaves))
    const traced = traceArgs(session, args, nums)
    const y = withTape(session.tape, () => f(...(traced.args as A)))
    const seed = scalarSeed('grad', y)
    const { cotangents } = sweep(session.tape, y, seed, traced.inputs, { createGraph: session.nested })
    const grads = unflatten(traced.flats, filled(cotangents, traced.inputs)).map((g) => settleTree(session, g))
    return { value: settle(session, y), grad: typeof argnums === 'number' ? grads[0] : grads }
  }
}

function settleTree(session: Session, tree: unknown): unknown {
  if (session.nested) return tree
  const flat = flattenTree(tree)
  return flat.rebuild(flat.leaves.map(unwrap))
}

/**
 * `grad(f)` is the gradient of a scalar function: `grad(f)(...args)` has the structure of argument `argnums`
 * (default 0). See `valueAndGrad` for the conventions. Nest it for higher derivatives: `grad(grad(f))`.
 *
 * @example grad((x: Value) => mul(x, sin(x)))(1) // sin 1 + cos 1
 */
export function grad<A extends unknown[]>(f: (...args: A) => Value): (...args: A) => A[0]
export function grad<A extends unknown[], N extends number>(
  f: (...args: A) => Value,
  options: { argnums: N },
): (...args: A) => A[N]
export function grad<A extends unknown[]>(
  f: (...args: A) => Value,
  options: { argnums: readonly number[] },
): (...args: A) => A[number][]
export function grad<A extends unknown[]>(f: (...args: A) => Value, options?: GradOptions): (...args: A) => unknown
export function grad<A extends unknown[]>(f: (...args: A) => Value, options: GradOptions = {}) {
  const both = valueAndGrad(f, options)
  return (...args: A) => both(...args).grad
}

// ── vjp, jvp, jacobian, hessian, hvp ─────────────────────────────────────────────────────────────────────────────────

/** The result of `vjp`: f's value and its pullback. */
export type VjpResult<T> = {
  value: number | Tensor
  /** Map a cotangent of the output (same kind and shape as `value`) to one of the input (the structure of x). */
  pullback: (cotangent: Value) => T
}

/**
 * The value of f at x and its pullback u ↦ uᵀJ, where J is the Jacobian of f at x. x is a number, tensor or pytree;
 * f returns a number or tensor. One forward pass; each call of `pullback` is one reverse pass.
 */
export function vjp<T>(f: (x: T) => Value, x: T): VjpResult<T> {
  const session = openSession(flattenTree(x).leaves)
  const traced = traceArgs(session, [x], [0])
  const y = withTape(session.tape, () => f(traced.args[0] as T))
  return {
    value: settle(session, y) as number | Tensor,
    pullback: (u) => {
      const { cotangents } = sweep(session.tape, y, u, traced.inputs, { createGraph: session.nested })
      return settleTree(session, unflatten(traced.flats, filled(cotangents, traced.inputs))[0]) as T
    },
  }
}

/** The result of `jvp`: f's value and the directional derivative J·v. */
export type JvpResult = { value: number | Tensor; tangent: number | Tensor }

/**
 * The value of f at x and the Jacobian–vector product J·v (the derivative of f at x in direction v), where v has the
 * structure of x and the result the shape of f(x).
 *
 * Method: aifn's primitives carry reverse-mode rules only, so forward mode is obtained by the transpose trick
 * ("forward from reverse"): the pullback u ↦ Jᵀu is linear in u, so J·v is the gradient with respect to u of
 * ⟨Jᵀu, v⟩. This costs one forward and two reverse passes, rather than the single pass of dual numbers
 * (Griewank and Walther, 2008, §3.1), and gives the same result to rounding.
 */
export function jvp<T>(f: (x: T) => Value, x: T, v: T): JvpResult {
  const session = openSession(flattenTree(x).leaves)
  const traced = traceArgs(session, [x], [0])
  const y = withTape(session.tape, () => f(traced.args[0] as T))
  const value = settle(session, y) as number | Tensor
  if (!isTraced(y)) return { value, tangent: zerosLike(unwrap(y)) }
  // u is a dummy cotangent: the reverse sweep with seed u records Jᵀu as a function of u.
  const u = session.tape.input(zerosLike(unwrap(y)), 'u')
  const { cotangents } = sweep(session.tape, y, u, traced.inputs, { createGraph: true })
  const directions = flattenTree(v).leaves
  if (directions.length !== traced.inputs.length) throw new Error('jvp: v must have the structure of x')
  const inner = withTape(session.tape, () => {
    let s: Value = 0
    cotangents.forEach((g, i) => {
      if (g !== null) s = add(s, sum(mul(g, directions[i])))
    })
    return s
  })
  const tangent = sweep(session.tape, inner, 1, [u], { createGraph: session.nested }).cotangents[0]
  return { value, tangent: (tangent === null ? zerosLike(unwrap(y)) : settle(session, tangent)) as number | Tensor }
}

/**
 * The Jacobian of f at x, for x a number or tensor and f returning a number or tensor: a tensor of shape
 * [...shape of f(x), ...shape of x] with entry [i, j] = ∂fᵢ/∂xⱼ (a number when both are numbers). Computed by one
 * reverse pass per output element, so it suits small outputs.
 */
export function jacobian(f: (x: Value) => Value): (x: Value) => Value {
  return (x) => {
    const { value, pullback } = vjpWithSession(f, x)
    const outShape = shapeOfValue(value)
    const inShape = shapeOfValue(x)
    const m = count(outShape)
    if (outShape.length === 0) return pullback(typeof unwrap(value) === 'number' ? 1 : ones([]))
    const rows: Value[] = []
    for (let i = 0; i < m; i++) {
      const e = zeros([m])
      ;(e.data as Float64Array)[i] = 1
      const row = pullback(reshape(e, outShape))
      rows.push(inShape.length === 0 ? row : reshape(row, [count(inShape)]))
    }
    // Inside another transform the rows are traced, and stacking them records on the enclosing (active) tape.
    return reshape(stack(rows), [...outShape, ...inShape])
  }
}

function vjpWithSession(f: (x: Value) => Value, x: Value) {
  const session = openSession([x])
  const traced = traceArgs(session, [x], [0])
  const y = withTape(session.tape, () => f(traced.args[0] as Value))
  return {
    value: y,
    pullback: (u: Value): Value => {
      const { cotangents } = sweep(session.tape, y, u, traced.inputs, { createGraph: session.nested })
      return settle(session, cotangents[0] ?? zerosLike(unwrap(x)))
    },
  }
}

/**
 * The Hessian of a scalar function f at x (a number or tensor): shape [...shape of x, ...shape of x], by reverse
 * over reverse (the Jacobian of the gradient). One reverse pass per element of x, so it suits small inputs.
 */
export function hessian(f: (x: Value) => Value): (x: Value) => Value {
  return jacobian(grad(f) as (x: Value) => Value)
}

/**
 * The Hessian–vector product H·v of a scalar function f at x, where v has the structure of x and so has the result.
 * Computed as the gradient of x ↦ ⟨∇f(x), v⟩ (reverse over reverse; Pearlmutter, 1994), at the cost of a few
 * gradient evaluations and without forming H.
 */
export function hvp<T>(f: (x: T) => Value, x: T, v: T): T {
  const directions = flattenTree(v).leaves
  const inner = (y: T): Value => {
    const g = flattenTree(grad(f)(y)).leaves
    if (g.length !== directions.length) throw new Error('hvp: v must have the structure of x')
    let s: Value = 0
    g.forEach((gi, i) => {
      s = add(s, sum(mul(gi, directions[i])))
    })
    return s
  }
  return grad(inner)(x)
}

// ── stopGradient ─────────────────────────────────────────────────────────────────────────────────────────────────────

const stopGradientOp = defineOp<undefined>(
  'stopGradient',
  ([x]) => x,
  () => [null],
)

/**
 * x itself, treated as a constant by every transform: its cotangent is zero. Applies to each leaf of a pytree. It is
 * recorded on the tape (so a graph shows where gradients stop).
 */
export function stopGradient<T>(x: T): T {
  const flat = flattenTree(x)
  return flat.rebuild(flat.leaves.map((leaf) => (isTraced(leaf) ? stopGradientOp([leaf], undefined) : leaf))) as T
}

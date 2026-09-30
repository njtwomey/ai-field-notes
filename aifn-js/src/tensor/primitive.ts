/**
 * Primitives: every operation is defined once, with its forward rule and its derivative rule (a vector–Jacobian
 * product). A primitive accepts numbers, tensors and traced values and returns the same kind: `exp(2)` is a number,
 * `exp(t)` a tensor, and `exp(traced)` a traced value recorded on its tape.
 *
 * - `defineUnary(name, f, df)` and `defineBinary(name, f, dfa, dfb)` make elementwise primitives from scalar rules.
 * - `defineOp(name, forward, vjp)` makes a general primitive (reductions, products, reshapes, slices).
 */

import { promote, scalarDType, type DType, type Tensor } from './core'
import { binaryDType, binaryKernel, ternaryKernel, unaryKernel } from './kernels'
import { currentTape, isTraced, unwrap, type Traced, type Value, type Vjp } from './tape'
import { mul } from './elementwise'
import { broadcastTo, sumTo } from './structure'
import { sum } from './reduce'

/** An untraced value: a number or a tensor. */
export type Raw = number | Tensor

/**
 * A general primitive's derivative rule: `vjp(cotangent, inputs, output, params)` returns one cotangent per input
 * (`null` for a zero cotangent). See `Vjp` for the conventions; `params` are the primitive's non-differentiable
 * arguments (axes, shapes, options).
 */
export type OpVjp<P> = (cotangent: Value, inputs: readonly Value[], output: Value, params: P) => (Value | null)[]

/** A general primitive: applies to a list of inputs and its parameters. */
export type Op<P> = (inputs: readonly Value[], params: P) => Value

/**
 * Define a general primitive. `forward(inputs, params)` computes the output from untraced inputs; `vjp` pulls a
 * cotangent back to the inputs and must be written with primitives (so that it can be traced again). Pass `vjp: null`
 * for an operation without a derivative: differentiating through it is then an error raised by the tape.
 *
 * @example
 * const cube = defineOp('cube', ([x]) => ..., (g, [x]) => [mul(g, mul(3, square(x)))])
 * cube([t], undefined)
 */
export function defineOp<P = undefined>(
  name: string,
  forward: (inputs: Raw[], params: P) => Raw,
  vjp: OpVjp<P> | null,
): Op<P> {
  return (inputs, params) => {
    let first: Traced | null = null
    for (const x of inputs) {
      if (isTraced(x)) {
        first = x
        break
      }
    }
    if (first === null) return forward(inputs as Raw[], params)
    const output = forward(inputs.map(unwrap), params)
    const rule: Vjp | null = vjp && ((g, ins, out) => vjp(g, ins, out, params))
    return (currentTape() ?? first.tape).record(name, inputs, output, rule)
  }
}

/** An elementwise function of one argument: numbers map to numbers, tensors to tensors, traced to traced. */
export interface Unary {
  (x: number): number
  (x: Tensor): Tensor
  (x: Traced): Traced
  (x: Value): Value
}

/** An elementwise function of two broadcast arguments. */
export interface Binary {
  (a: number, b: number): number
  (a: Tensor, b: Tensor | number): Tensor
  (a: number, b: Tensor): Tensor
  (a: Traced, b: Value): Traced
  (a: Value, b: Traced): Traced
  (a: Value, b: Value): Value
}

/** Options for elementwise primitives. */
export type ElementwiseOptions = {
  /**
   * The result of integer inputs is an integer (as for negation or addition), so int32 tensors stay int32. By default
   * int32 inputs give float64 results, as for exp or division.
   */
  integer?: boolean
}

/** The derivative dy/dx of a unary rule at x, given y = f(x). */
export type UnaryDerivative = (x: number, y: number) => number

/** A partial derivative ∂y/∂a or ∂y/∂b of a binary rule at (a, b), given y = f(a, b). */
export type BinaryDerivative = (a: number, b: number, y: number) => number

/** Output dtype of an elementwise map of a tensor. */
function unaryDType(t: Tensor, integer: boolean | undefined): DType {
  return t.dtype === 'int32' && !integer ? 'float64' : t.dtype
}

/**
 * Reduce a cotangent `v` (shaped like a broadcast result) to the kind and shape of `like`: summed over broadcast axes,
 * and summed to a number when `like` is a number.
 */
export function sumLike(v: Value, like: Value): Value {
  const target = unwrap(like)
  const value = unwrap(v)
  if (typeof target === 'number') return typeof value === 'number' ? v : sum(v as Tensor)
  if (typeof value === 'number') return broadcastTo(v, target.shape)
  const same = value.shape.length === target.shape.length && value.shape.every((d, k) => d === target.shape[k])
  return same ? v : sumTo(v, target.shape)
}

/**
 * Make an elementwise unary primitive from a forward rule and a vjp written with primitives:
 * `rule(g, x, y)` returns the cotangent of x (shaped like y). For modules defining built-ins with higher derivatives.
 */
export function unaryPrimitive(
  name: string,
  f: (x: number) => number,
  rule: ((g: Value, x: Value, y: Value) => Value | null) | null,
  options: ElementwiseOptions = {},
): Unary {
  const op = defineOp<undefined>(
    name,
    ([x]) => (typeof x === 'number' ? f(x) : unaryKernel(x, f, unaryDType(x, options.integer))),
    rule && ((g, [x], y) => [rule(g, x, y)]),
  )
  return ((x: Value) => op([x], undefined)) as Unary
}

/**
 * Make an elementwise binary primitive from a forward rule and a vjp written with primitives: `rule(g, a, b, y)`
 * returns the cotangents of a and b before reduction over broadcast axes (shaped like y); the reduction is applied
 * here. `null` in place of the rule means no derivative at all.
 */
export function binaryPrimitive(
  name: string,
  f: (a: number, b: number) => number,
  rule: ((g: Value, a: Value, b: Value, y: Value) => [Value | null, Value | null]) | null,
  options: ElementwiseOptions = {},
): Binary {
  const op = defineOp<undefined>(
    name,
    ([a, b]) => {
      if (typeof a === 'number' && typeof b === 'number') return f(a, b)
      const dtype = binaryDType(a, b)
      return binaryKernel(a, b, f, dtype === 'int32' && !options.integer ? 'float64' : dtype)
    },
    rule &&
      ((g, [a, b], y) => {
        const [ga, gb] = rule(g, a, b, y)
        return [ga === null ? null : sumLike(ga, a), gb === null ? null : sumLike(gb, b)]
      }),
  )
  return ((a: Value, b: Value) => op([a, b], undefined)) as Binary
}

/**
 * Define an elementwise primitive of one argument from its scalar rule `f` and derivative `df(x, y)` = dy/dx at x,
 * where y = f(x). Pass `df: null` when there is no derivative. The result maps numbers to numbers and tensors to
 * tensors elementwise (int32 inputs give float64 unless `options.integer`), and records itself when traced.
 *
 * The scalar `df` gives first derivatives. For second derivatives, pass `options.derivative(x, y)`: dy/dx written
 * with primitives (e.g. `(x, y) => mul(y, sub(1, y))` for the logistic sigmoid). Without it, a second derivative goes
 * through a derived primitive `name′` that has no derivative rule, which is an error rather than a silent zero.
 *
 * @example const softplus = defineUnary('softplus', (x) => Math.log1p(Math.exp(x)), (x) => 1 / (1 + Math.exp(-x)))
 */
export function defineUnary(
  name: string,
  f: (x: number) => number,
  df: UnaryDerivative | null,
  options: ElementwiseOptions & { derivative?: (x: Value, y: Value) => Value } = {},
): Unary {
  if (df === null) return unaryPrimitive(name, f, null, options)
  let derivative: Unary | null = null
  return unaryPrimitive(
    name,
    f,
    (g, x, y) => {
      const rx = unwrap(x)
      const ry = unwrap(y)
      let d: Value
      if (isTraced(x) && options.derivative) d = options.derivative(x, y)
      else if (isTraced(x)) {
        derivative ??= defineUnary(`${name}′`, (v) => df(v, f(v)), null)
        d = derivative(x)
      } else if (typeof rx === 'number') d = df(rx, ry as number)
      else d = binaryKernel(rx, ry as Tensor, df, 'float64')
      return mul(g, d)
    },
    options,
  )
}

/** A partial derivative ∂y/∂a or ∂y/∂b of a binary primitive at (a, b), given y = f(a, b), written with primitives. */
export type BinaryPartial = (a: Value, b: Value, y: Value) => Value

/**
 * The cotangent of argument `which` of an elementwise primitive with scalar partials `partials` (one per argument,
 * each `(...args, y) => ∂y/∂arg`) and optional primitive partials `derivative`. Shared by `defineBinary` and
 * `defineTernary`; the reduction over broadcast axes is applied by the caller.
 */
function elementwisePartial(
  name: string,
  f: (...args: number[]) => number,
  partials: readonly (((...args: number[]) => number) | null)[],
  derivative: readonly (((...args: Value[]) => Value) | null)[] | undefined,
  derived: (Op<undefined> | null)[],
  which: number,
  g: Value,
  args: readonly Value[],
  y: Value,
): Value | null {
  const df = partials[which]
  const input = args[which]
  if (df === null) {
    if (isTraced(input)) throw new Error(`${name}: no derivative with respect to argument ${which + 1}`)
    return null
  }
  const anyTraced = args.some(isTraced)
  const rule = derivative?.[which]
  if (anyTraced && rule) return mul(g, rule(...args, y))
  // The derivative computed from the scalar partial. When traced, it goes through a derived primitive `name′ₖ`
  // without a derivative of its own, so a second derivative is an error rather than a silent zero.
  const at = (...v: number[]) => df(...v, f(...v))
  let d: Value
  if (anyTraced) {
    derived[which] ??= defineOp<undefined>(`${name}′${'abc'[which]}`, (raw) => elementwiseRaw(raw, at, false), null)
    d = derived[which]([...args], undefined)
  } else d = elementwiseRaw(args.map(unwrap), at, false)
  return mul(g, d)
}

/**
 * The result dtype of an elementwise rule of several raw arguments: tensors' dtypes promote, numbers are weak scalars
 * (as in `binaryDType`), and an int32 result becomes float64 unless the rule maps integers to integers.
 */
function elementwiseDType(args: readonly Raw[], integer: boolean | undefined): DType {
  let dtype: DType | null = null
  for (const v of args) if (typeof v !== 'number') dtype = dtype === null ? v.dtype : promote(dtype, v.dtype)
  if (dtype === null) return 'float64'
  for (const v of args) if (typeof v === 'number') dtype = scalarDType(v, dtype)
  return dtype === 'int32' && !integer ? 'float64' : dtype
}

/** Apply a scalar rule of any arity elementwise to broadcast raw arguments: numbers give a number. */
function elementwiseRaw(args: readonly Raw[], f: (...v: number[]) => number, integer: boolean | undefined): Raw {
  if (args.every((v) => typeof v === 'number')) return f(...(args as number[]))
  const dtype = elementwiseDType(args, integer)
  if (args.length === 1) return unaryKernel(args[0] as Tensor, f, dtype)
  if (args.length === 2) return binaryKernel(args[0], args[1], f, dtype)
  if (args.length === 3) return ternaryKernel(args[0], args[1], args[2], f, dtype)
  throw new Error(`elementwise: arity ${args.length} is not supported`)
}

/**
 * Define an elementwise primitive of two broadcast arguments from its scalar rule `f` and partial derivatives
 * `dfa(a, b, y)` = ∂y/∂a and `dfb(a, b, y)` = ∂y/∂b, where y = f(a, b). A `null` partial means the primitive cannot
 * be differentiated in that argument: doing so throws. Cotangents are summed over broadcast axes.
 *
 * The scalar partials give first derivatives. For second derivatives, pass `options.derivative`: the two partials
 * written with primitives, `[(a, b, y) => ∂y/∂a, (a, b, y) => ∂y/∂b]` (`null` for an argument whose partial is only
 * available as the scalar rule). Without it, differentiating a derivative is an error rather than a silent zero.
 *
 * @example const hypot = defineBinary('hypot', Math.hypot, (a, _b, y) => a / y, (_a, b, y) => b / y,
 *   { derivative: [(a, _b, y) => div(a, y), (_a, b, y) => div(b, y)] })
 */
export function defineBinary(
  name: string,
  f: (a: number, b: number) => number,
  dfa: BinaryDerivative | null,
  dfb: BinaryDerivative | null,
  options: ElementwiseOptions & { derivative?: readonly [BinaryPartial | null, BinaryPartial | null] } = {},
): Binary {
  const derived: (Op<undefined> | null)[] = [null, null]
  const partials = [dfa, dfb] as (((...args: number[]) => number) | null)[]
  const derivative = options.derivative as readonly (((...args: Value[]) => Value) | null)[] | undefined
  const g2 = f as (...args: number[]) => number
  return binaryPrimitive(
    name,
    f,
    dfa === null && dfb === null
      ? null
      : (g, a, b, y) => [
          elementwisePartial(name, g2, partials, derivative, derived, 0, g, [a, b], y),
          elementwisePartial(name, g2, partials, derivative, derived, 1, g, [a, b], y),
        ],
    options,
  )
}

/** An elementwise function of three broadcast arguments: numbers give a number, tensors a tensor, traced a traced. */
export interface Ternary {
  (a: number, b: number, c: number): number
  (a: Traced, b: Value, c: Value): Traced
  (a: Value, b: Traced, c: Value): Traced
  (a: Value, b: Value, c: Traced): Traced
  (a: Tensor, b: Raw, c: Raw): Tensor
  (a: Raw, b: Tensor, c: Raw): Tensor
  (a: Raw, b: Raw, c: Tensor): Tensor
  (a: Value, b: Value, c: Value): Value
}

/** A partial derivative of a ternary rule at (a, b, c), given y = f(a, b, c). */
export type TernaryDerivative = (a: number, b: number, c: number, y: number) => number

/** A partial derivative of a ternary primitive at (a, b, c), given y = f(a, b, c), written with primitives. */
export type TernaryPartial = (a: Value, b: Value, c: Value, y: Value) => Value

/**
 * Define an elementwise primitive of three broadcast arguments from its scalar rule `f` and its partial derivatives
 * `dfa`, `dfb`, `dfc` (each `(a, b, c, y) => ∂y/∂·`, `null` where there is none: differentiating that argument
 * throws). Cotangents are summed over broadcast axes. `options.derivative` gives the partials written with primitives,
 * for second derivatives, as for `defineBinary`.
 *
 * @example const fma = defineTernary('fma', (a, b, c) => a * b + c, (_a, b) => b, (a) => a, () => 1)
 */
export function defineTernary(
  name: string,
  f: (a: number, b: number, c: number) => number,
  dfa: TernaryDerivative | null,
  dfb: TernaryDerivative | null,
  dfc: TernaryDerivative | null,
  options: ElementwiseOptions & {
    derivative?: readonly [TernaryPartial | null, TernaryPartial | null, TernaryPartial | null]
  } = {},
): Ternary {
  const derived: (Op<undefined> | null)[] = [null, null, null]
  const partials = [dfa, dfb, dfc] as (((...args: number[]) => number) | null)[]
  const derivative = options.derivative as readonly (((...args: Value[]) => Value) | null)[] | undefined
  const g3 = f as (...args: number[]) => number
  const op = defineOp<undefined>(
    name,
    (args) => elementwiseRaw(args, g3, options.integer),
    partials.every((p) => p === null)
      ? null
      : (g, args, y) =>
          args.map((input, i) => {
            const gi = elementwisePartial(name, g3, partials, derivative, derived, i, g, args, y)
            return gi === null ? null : sumLike(gi, input)
          }),
  )
  return ((a: Value, b: Value, c: Value) => op([a, b, c], undefined)) as Ternary
}

/** The result kind of a tensor-valued primitive of `X`: traced when `X` is traced, otherwise a tensor. */
export type TensorResult<X> = X extends Traced ? Traced : Tensor

/** The result kind of a number-valued primitive of `X` (e.g. a full reduction): traced or a number. */
export type NumberResult<X> = X extends Traced ? Traced : number

/** The result kind of a primitive of two inputs whose untraced result is `R`. */
export type Result2<A, B, R> = A extends Traced ? Traced : B extends Traced ? Traced : R

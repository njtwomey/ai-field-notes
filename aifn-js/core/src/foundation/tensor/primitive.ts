/**
 * Primitives: every operation is defined once, with its forward rule and its derivative rule (a vector–Jacobian
 * product). A primitive accepts numbers, tensors and traced values and returns the same kind: `exp(2)` is a number,
 * `exp(t)` a tensor, and `exp(traced)` a traced value recorded on its tape.
 *
 * - `elementwise({ id, f, derivative })` makes an elementwise primitive from its scalar rule and one derivative per
 *   argument, written with primitives.
 * - `defineOp(name, forward, vjp)` makes a general primitive (reductions, products, reshapes, slices).
 */

import { promote, scalarDType, type DType, type Tensor } from './core'
import { AifnError, NotDifferentiableError } from 'aifn/foundation/errors'
import { binaryDType, binaryKernel, ternaryKernel, unaryKernel, type Arithmetic } from './kernels'
import {
  parseId,
  register,
  type DTypeRule,
  type Op,
  type OpVjp,
  type PrimitiveDoc,
  type PrimitiveTest,
  type Raw,
} from './registry'
import { currentTape, isTraced, unwrap, type Traced, type Value, type Vjp } from './tape'
import { mul } from './elementwise'
import { broadcastTo, sumTo } from './structure'
import { sum } from './reduce'

export type { Op, OpVjp, Raw }

/** The specification of a primitive for `definePrimitive`. */
export type PrimitiveSpec<P> = {
  /**
   * `module/name`, e.g. `numerics/linalg/cholesky`: registered, and unique (a second registration throws). A bare name defines
   * a local primitive that is not registered (see `registry`).
   */
  readonly id: string
  /** Number of inputs, or `variadic` for a list. */
  readonly arity?: number | 'variadic'
  /** The forward rule on untraced inputs (numbers and tensors). */
  readonly impl: (inputs: Raw[], params: P) => Raw
  /**
   * The vector–Jacobian product, written with primitives so that it can be traced again; `null` for an operation
   * without a derivative (differentiating through it is then an error raised by the tape).
   */
  readonly vjp: OpVjp<P> | null
  /** Which inputs the rule differentiates (default: all when `vjp` is given). */
  readonly differentiable?: readonly boolean[] | boolean
  readonly kind?: 'elementwise' | 'general'
  /** Elementwise primitives: the result dtype of int32 inputs. */
  readonly dtype?: DTypeRule
  readonly doc?: PrimitiveDoc
  readonly test?: PrimitiveTest
}

/**
 * Define a primitive: its forward rule `impl` and its derivative rule `vjp`, registered under `id` (design K §5). The
 * result applies the primitive to a list of inputs (numbers, tensors or traced values) and its parameters: untraced
 * inputs go straight to `impl`; with a traced input the application is recorded on the tape with its rule.
 *
 * @example
 * const cubeOp = definePrimitive({ id: 'demo/cube', arity: 1, impl: ([x]) => ..., vjp: (g, [x]) => [mul(g, mul(3, square(x)))] })
 * cubeOp([t], undefined)
 */
export function definePrimitive<P = undefined>(spec: PrimitiveSpec<P>): Op<P> {
  const parsed = parseId(spec.id)
  const name = parsed?.name ?? spec.id
  const { impl, vjp } = spec
  // The fast path (design K §3.5): with no traced input the application is the raw forward rule. A number is never
  // traced, so only objects are checked.
  const apply: Op<P> = (inputs, params) => {
    let first: Traced | null = null
    for (let k = 0; k < inputs.length; k++) {
      const x = inputs[k]
      if (typeof x !== 'number' && isTraced(x)) {
        first = x
        break
      }
    }
    if (first === null) return impl(inputs as Raw[], params)
    const output = impl(inputs.map(unwrap), params)
    const rule: Vjp | null = vjp && ((g, ins, out) => vjp(g, ins, out, params))
    return (currentTape() ?? first.tape).record(name, inputs, output, rule)
  }
  if (parsed) {
    register<P>({
      id: spec.id,
      module: parsed.module,
      name,
      kind: spec.kind ?? 'general',
      arity: spec.arity ?? 'variadic',
      apply,
      impl,
      differentiable: spec.differentiable ?? vjp !== null,
      dtype: spec.dtype,
      doc: spec.doc ?? {},
      test: spec.test ?? {},
    })
  }
  return apply
}

/** Registry metadata a `define*` wrapper may carry: documentation and test domains. */
export type PrimitiveMeta = { readonly doc?: PrimitiveDoc; readonly test?: PrimitiveTest }

/**
 * Define a general primitive (a thin wrapper over `definePrimitive`). `forward(inputs, params)` computes the output
 * from untraced inputs; `vjp` pulls a cotangent back to the inputs and must be written with primitives (so that it can
 * be traced again). Pass `vjp: null` for an operation without a derivative: differentiating through it is then an
 * error raised by the tape. `name` is the registry id, `module/name`; the part after the slash is recorded on tapes.
 *
 * @example
 * const cube = defineOp('demo/cube', ([x]) => ..., (g, [x]) => [mul(g, mul(3, square(x)))])
 * cube([t], undefined)
 */
export function defineOp<P = undefined>(
  name: string,
  forward: (inputs: Raw[], params: P) => Raw,
  vjp: OpVjp<P> | null,
  meta: PrimitiveMeta & { readonly arity?: number | 'variadic'; readonly differentiable?: readonly boolean[] } = {},
): Op<P> {
  return definePrimitive<P>({ id: name, impl: forward, vjp, ...meta })
}

/** The dtype rule of `ElementwiseOptions`. */
function dtypeRule(options: ElementwiseOptions): DTypeRule {
  return options.integer ? 'same' : 'float'
}

/** The short name of an id (`foundation/tensor/exp` → `exp`), for error messages and derived names. */
function shortName(id: string): string {
  return id.slice(id.lastIndexOf('/') + 1)
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
  /** Documentation for the registry (design K §5). */
  doc?: PrimitiveDoc
  /** Test domains and tolerances for the generated primitive tests. */
  test?: PrimitiveTest
}

/**
 * The registry metadata of an elementwise primitive of `arity` arguments. `secondOrder` says whether its derivative is
 * written with primitives (so second derivatives exist); the test metadata may override it.
 */
function elementwiseMeta(
  arity: number,
  options: ElementwiseOptions,
  differentiable: readonly boolean[] | boolean,
  secondOrder: boolean,
) {
  return {
    kind: 'elementwise' as const,
    arity,
    dtype: dtypeRule(options),
    differentiable,
    doc: options.doc,
    test: { secondOrder, ...options.test },
  }
}

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
  options: ElementwiseOptions & { secondOrder?: boolean } = {},
): Unary {
  const op = definePrimitive<undefined>({
    id: name,
    impl: ([x]) => (typeof x === 'number' ? f(x) : unaryKernel(x, f, unaryDType(x, options.integer))),
    vjp: rule && ((g, [x], y) => [rule(g, x, y)]),
    ...elementwiseMeta(1, options, rule !== null, options.secondOrder ?? rule !== null),
  })
  // Scalar fast path: a number is never traced, so it goes straight to the scalar rule.
  return ((x: Value) => (typeof x === 'number' ? f(x) : op([x], undefined))) as Unary
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
  options: ElementwiseOptions & {
    differentiable?: readonly boolean[]
    secondOrder?: boolean
    /** An arithmetic operation with a dedicated kernel loop (see `Arithmetic` in kernels.ts). */
    kernel?: Arithmetic
  } = {},
): Binary {
  const op = definePrimitive<undefined>({
    id: name,
    impl: ([a, b]) => {
      if (typeof a === 'number' && typeof b === 'number') return f(a, b)
      const dtype = binaryDType(a, b)
      return binaryKernel(a, b, f, dtype === 'int32' && !options.integer ? 'float64' : dtype, options.kernel)
    },
    vjp:
      rule &&
      ((g, [a, b], y) => {
        const [ga, gb] = rule(g, a, b, y)
        return [ga === null ? null : sumLike(ga, a), gb === null ? null : sumLike(gb, b)]
      }),
    ...elementwiseMeta(2, options, options.differentiable ?? rule !== null, options.secondOrder ?? rule !== null),
  })
  // Scalar fast path: numbers are never traced, so two numbers go straight to the scalar rule.
  return ((a: Value, b: Value) =>
    typeof a === 'number' && typeof b === 'number' ? f(a, b) : op([a, b], undefined)) as Binary
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
  throw new AifnError('elementwise', `elementwise: arity ${args.length} is not supported`)
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

/** A derivative of an elementwise primitive with respect to one argument, written with primitives: `(...args, y)`. */
export type ElementwiseDerivative = (...argsAndOutput: Value[]) => Value

/** The specification of an elementwise primitive for `elementwise`. */
export type ElementwiseSpec = {
  /** `module/name`, registered (see `definePrimitive`). */
  readonly id: string
  /** The scalar rule. Its arity (1, 2 or 3) is the primitive's. */
  readonly f: (...x: number[]) => number
  /**
   * One derivative per argument, ∂y/∂xᵢ as a function of the arguments and the output y, written with primitives so
   * that it can be differentiated again; `null` where the primitive is not differentiable in that argument (doing so
   * throws `NotDifferentiableError`).
   */
  readonly derivative: readonly (ElementwiseDerivative | null)[]
  /** The result dtype of int32 inputs: `same` (int32) or `float` (float64, the default). */
  readonly dtype?: DTypeRule
  readonly doc?: PrimitiveDoc
  readonly test?: PrimitiveTest
}

/**
 * Define an elementwise primitive from its scalar rule and **one** derivative per argument (design K §4.2): the vjp is
 * g·∂y/∂xᵢ summed over broadcast axes, and it is differentiable to any order because the derivatives are primitives.
 * Numbers give numbers, tensors broadcast (NumPy rules), and traced inputs are recorded.
 *
 * @example
 * const cube = elementwise({ id: 'demo/cube', f: (x) => x ** 3, derivative: [(x) => mul(3, square(x))] })
 */
export function elementwise(spec: ElementwiseSpec & { readonly f: (x: number) => number }): Unary
export function elementwise(spec: ElementwiseSpec & { readonly f: (a: number, b: number) => number }): Binary
export function elementwise(spec: ElementwiseSpec): Ternary
export function elementwise(spec: ElementwiseSpec): Unary | Binary | Ternary {
  const arity = spec.derivative.length
  if (arity < 1 || arity > 3 || spec.f.length > arity) {
    throw new AifnError(
      'elementwise',
      `elementwise: ${spec.id} needs one derivative per argument (1 to 3), got ${arity} for f of arity ${spec.f.length}`,
    )
  }
  const integer = spec.dtype === 'same'
  const name = shortName(spec.id)
  const op = definePrimitive<undefined>({
    id: spec.id,
    kind: 'elementwise',
    arity,
    dtype: spec.dtype ?? 'float',
    impl: (args) => elementwiseRaw(args, spec.f, integer),
    vjp: spec.derivative.every((d) => d === null)
      ? null
      : (g, args, y) =>
          args.map((input, i) => {
            const d = spec.derivative[i]
            if (d === null) {
              if (isTraced(input)) {
                throw new NotDifferentiableError(name, `${name}: no derivative with respect to argument ${i + 1}`, i)
              }
              return null
            }
            return sumLike(mul(g, d(...args, y)), input)
          }),
    differentiable: spec.derivative.map((d) => d !== null),
    doc: spec.doc,
    test: spec.test,
  })
  // Scalar fast paths: numbers are never traced, so all-number arguments go straight to the scalar rule.
  const f = spec.f
  if (arity === 1) return ((x: Value) => (typeof x === 'number' ? f(x) : op([x], undefined))) as Unary
  if (arity === 2)
    return ((a: Value, b: Value) =>
      typeof a === 'number' && typeof b === 'number' ? f(a, b) : op([a, b], undefined)) as Binary
  return ((a: Value, b: Value, c: Value) =>
    typeof a === 'number' && typeof b === 'number' && typeof c === 'number'
      ? f(a, b, c)
      : op([a, b, c], undefined)) as Ternary
}

/** The result kind of a tensor-valued primitive of `X`: traced when `X` is traced, otherwise a tensor. */
export type TensorResult<X> = X extends Traced ? Traced : Tensor

/** The result kind of a number-valued primitive of `X` (e.g. a full reduction): traced or a number. */
export type NumberResult<X> = X extends Traced ? Traced : number

/** The result kind of a primitive of two inputs whose untraced result is `R`. */
export type Result2<A, B, R> = A extends Traced ? Traced : B extends Traced ? Traced : R

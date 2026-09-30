/**
 * Numbers: the value types every signature in aifn is written in, and the input aliases that relax them.
 *
 * Mathematical values are `Scalar` (or `Scalar | Tensor`); integer metadata (counts, positions, axes, shapes) has its
 * own names, so a signature never confuses a value with a count: `erf(x: Scalar | Tensor)`, `zeros(shape: Shape)`,
 * `sum(x, axis?: Axes)`. Every name here is a type; nothing is emitted at run time.
 */

// ── Scalars and integer metadata ─────────────────────────────────────────────────────────────────────────────────────

/** A real number used as a mathematical value (a parameter, an observation, a result). An alias of `number`. */
export type Scalar = number

/** A non-negative integer count or length: a number of elements, rows, steps or draws. */
export type Size = number

/** A zero-based integer position along an axis or in a list. Negative indices count from the end where documented. */
export type Index = number

/** One axis of a tensor, zero-based; negative axes count from the end (−1 is the last). */
export type Axis = number

/** One axis or several (duplicates are an error); functions that reduce read `null` or omission as every axis. */
export type Axes = Axis | readonly Axis[]

/** The length of each axis of a tensor; `[]` for a scalar tensor. */
export type Shape = readonly Size[]

// ── Tensor ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Element type of a tensor's storage today. The target adds `bool` and interleaved `complex128` (see `DTypeTarget`). */
export type DType = 'float64' | 'float32' | 'int32'

/**
 * The target element types (design K §3.1): `bool` (Uint8Array) for masks and comparisons, and `complex128` stored as
 * interleaved real and imaginary parts in a Float64Array. TODO(phase 1): becomes `DType`.
 */
export type DTypeTarget = 'bool' | 'int32' | 'float32' | 'float64' | 'complex128'

/** The typed array that backs a tensor. */
export type TensorData = Float64Array | Float32Array | Int32Array

declare const tensorBrand: unique symbol

/**
 * The type of the symbol that brands a tensor. The runtime symbol is `aifn/foundation/tensor`'s `TENSOR`
 * (`Symbol.for('aifn.tensor')`), set only by its constructors and typed as this, so this `Tensor` and `aifn/foundation/tensor`'s
 * are one type.
 */
export type TensorBrand = typeof tensorBrand

/**
 * An n-dimensional array: `shape[k]` elements along axis k, stored in `data` at `offset + Σ_k index[k] · strides[k]`
 * (NumPy's strided layout). Immutable by convention: operations return new tensors, and views share `data`. Only
 * `aifn/foundation/tensor` constructs tensors.
 */
export interface Tensor {
  /** The brand set by `aifn/foundation/tensor`'s constructors; `isTensor` checks it (no duck typing). */
  readonly [tensorBrand]: true
  /** Length of each axis; `[]` for a scalar tensor. */
  readonly shape: Shape
  /** Step in `data`, in elements, for a unit step along each axis. Row-major (C order) by default. */
  readonly strides: readonly number[]
  /** Position in `data` of the element at index (0, …, 0). */
  readonly offset: Index
  readonly dtype: DType
  readonly data: TensorData
}

/** A rank-1 tensor (documentation only: the rank is checked at run time). */
export type Vector = Tensor

/** A rank-2 tensor (documentation only: the rank is checked at run time). */
export type Matrix = Tensor

/**
 * A tensor on the wire (JSON, fixtures, aifn-py): dtype, shape and row-major data, with non-finite values written as
 * the strings `"nan"`, `"inf"` and `"-inf"`.
 */
export interface TensorWire {
  readonly dtype: DType
  readonly shape: Shape
  readonly data: readonly (number | 'nan' | 'inf' | '-inf')[]
}

// ── Traced values and the tape protocol ──────────────────────────────────────────────────────────────────────────────

declare const tracedBrand: unique symbol

/**
 * The type of the symbol that marks a traced value. The runtime symbol is `Symbol.for('aifn.traced')`, created in
 * `aifn/foundation/tensor` and typed as this.
 */
export type TracedBrand = typeof tracedBrand

/** A number or tensor recorded on a tape: `value` is the underlying data and `id` its node on `tape`. */
export interface Traced<T extends Scalar | Tensor = Scalar | Tensor> {
  readonly [tracedBrand]: true
  readonly value: T
  readonly id: Index
  readonly tape: Tape
}

/** An untraced value: a number or a tensor. What a primitive computes on. */
export type Raw = Scalar | Tensor

/** Anything a primitive accepts: a number, a tensor, or either of them traced. */
export type Value = Scalar | Tensor | Traced

/**
 * A vector–Jacobian product rule: given the cotangent of the output (same kind and shape as the output), the inputs and
 * the output, return one cotangent per input, each of the same kind and shape as its input; `null` is a zero
 * cotangent. Rules are written with primitives, so they can be traced again.
 */
export type Vjp = (cotangent: Value, inputs: readonly Value[], output: Value) => (Value | null)[]

/** Records primitive applications. Implemented by `aifn/foundation/autodiff`. */
export interface Tape {
  /**
   * Record that primitive `name` mapped `inputs` (as passed, some traced) to `output` (the raw forward value), and
   * return the traced output. `vjp` is `null` for a primitive without a derivative rule: differentiating through it
   * must then be an error, never a silent zero.
   */
  record(name: string, inputs: readonly Value[], output: Raw, vjp: Vjp | null): Traced
}

// ── Inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A number or a tensor, as elementwise operations accept: a number broadcasts as a scalar. The same type as `Raw`,
 * named for inputs.
 */
export type TensorLike = Raw

/** Nested arrays of numbers, as `tensor` accepts and `toArray` returns. */
export type NestedArray = Scalar | NestedArray[]

/**
 * A vector argument: a rank-1 tensor (any strides) or a plain or typed array of numbers. Always copied on the way in.
 * Accepted wherever a function reads one vector of data (a point, coefficients, weights, a signal).
 */
export type VectorLike = Tensor | ArrayLike<Scalar>

/**
 * A matrix argument: a rank-2 tensor or rows of numbers (all of one length). Always copied on the way in. Accepted
 * wherever a function reads one matrix (a system matrix, a design matrix, rows of points).
 */
export type MatrixLike = Tensor | ArrayLike<ArrayLike<Scalar>>

/**
 * Numeric data read flat: a tensor of any rank (read row-major) or an array of numbers. Accepted by statistics and
 * metrics, which treat their input as a sample rather than as a vector.
 */
export type DataLike = ArrayLike<Scalar> | Tensor

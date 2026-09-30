/**
 * `aifn/tensor`: the n-dimensional array everything numeric builds on, and the primitive machinery that defines each
 * operation once with its derivative rule.
 *
 * - Type and storage: `Tensor` (shape, strides, offset, dtype, typed-array data), `Vector`, `Matrix`, `DType`.
 * - Constructors: `tensor`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `fromRows`, `scalar`, `fromData`.
 * - Converters (the chart boundary): `toArray`, `toRows`, `toFlat`, `item`.
 * - Primitives (numbers, tensors and traced values alike): elementwise, structural, reductions and products.
 * - Defining primitives: `defineUnary`, `defineBinary`, `defineTernary`, `defineOp`, `sumLike`; the autodiff hook: `Tape`, `withTape`, `traced`.
 * - Indexed reads and writes (primitives): `gather`, `scatterAdd`, `take` (rows along the first axis).
 * - Inputs: `VectorLike` and `MatrixLike`, the one sanctioned relaxation of `Tensor` for data arguments.
 * - Dense kernels for inner loops, on row-major `Float64Array`s: the `dense` namespace (`dense.toF64`,
 *   `dense.toMatrixF64`, `dense.dot`, `dense.matVec`, `dense.matMul`, `dense.axpy`, …). Not primitives.
 */

export type { Axis, DType, NestedArray, Tensor, TensorData, TensorLike, Vector, Matrix } from './core'
export { fromData, isContiguous, isTensor, rowMajorStrides, size } from './core'
export {
  arange,
  astype,
  copy,
  eye,
  fromRows,
  full,
  item,
  linspace,
  ones,
  scalar,
  shapeOf,
  tensor,
  toArray,
  toFlat,
  toRows,
  zeros,
} from './create'
export { broadcastShapes, type SliceSpec } from './views'
export { currentTape, isTraced, traced, unwrap, withTape, type Tape, type Traced, type Value, type Vjp } from './tape'
export {
  defineBinary,
  defineOp,
  defineTernary,
  defineUnary,
  sumLike,
  type Binary,
  type BinaryDerivative,
  type BinaryPartial,
  type ElementwiseOptions,
  type NumberResult,
  type Op,
  type OpVjp,
  type Raw,
  type Result2,
  type TensorResult,
  type Ternary,
  type TernaryDerivative,
  type TernaryPartial,
  type Unary,
  type UnaryDerivative,
} from './primitive'
export {
  abs,
  add,
  clip,
  cos,
  div,
  equalTo,
  exp,
  expm1,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  log1p,
  map,
  map2,
  maximum,
  minimum,
  mul,
  neg,
  notEqualTo,
  pow,
  sign,
  sin,
  sqrt,
  square,
  sub,
  tanh,
  where,
  type Comparison,
} from './elementwise'
export {
  broadcastTo,
  concat,
  diag,
  diagonal,
  expandDims,
  flatten,
  get,
  permute,
  reshape,
  set,
  shapeOfValue,
  slice,
  squeeze,
  stack,
  sumTo,
  transpose,
} from './structure'
export { argmax, argmin, logsumexp, max, mean, min, norm, prod, std, sum, variance, type Reduction } from './reduce'
export { dot, einsum, matmul, outer } from './products'
export { allclose, equal, type CloseOptions } from './compare'
export { gather, scatterAdd, take } from './gather'
export type { MatrixLike, VectorLike } from './dense'
export * as dense from './dense'

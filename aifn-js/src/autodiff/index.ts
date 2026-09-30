/**
 * `aifn/autodiff`: reverse-mode automatic differentiation over aifn's primitives.
 *
 * Every operation in aifn is a primitive with its derivative rule (`aifn/tensor`'s `defineUnary`, `defineBinary`,
 * `defineTernary`, `defineOp`). This module supplies the tape those primitives record on and the transforms built on
 * it; it defines no operations of its own. A function differentiated here is ordinary code over numbers and tensors:
 * the same `softplus`, `matmul` or `cholesky` that computes values records itself when given a traced value.
 *
 * - Transforms: `grad`, `valueAndGrad` (with `argnums`), `vjp`, `jvp`, `jacobian`, `hessian`, `hvp`, `stopGradient`.
 *   Arguments may be numbers, tensors or pytrees (nested arrays and plain objects) of them. Transforms nest:
 *   `grad(grad(f))` is a second derivative wherever the primitives involved declare `options.derivative`; a primitive
 *   whose derivative has no rule of its own makes a second derivative an error, never a silent zero.
 * - Checking: `gradCheck(f, x)` compares gradients with central finite differences and reports every element.
 * - Inspection: `traceGraph(f, x)` returns the recorded nodes with values and adjoints in topological order.
 *
 * Forward mode (`jvp`) is derived from reverse mode by the transpose trick rather than dual numbers, since primitives
 * carry vjp rules only; see `jvp`.
 */

export { gradCheck, type GradCheckEntry, type GradCheckOptions, type GradCheckReport } from './check'
export { traceGraph, type Graph, type GraphInput, type GraphNode } from './graph'
export { NotDifferentiableError } from './tape'
export {
  grad,
  hessian,
  hvp,
  jacobian,
  jvp,
  stopGradient,
  valueAndGrad,
  vjp,
  type GradOptions,
  type JvpResult,
  type ValueAndGrad,
  type VjpResult,
} from './transforms'
export { type Tree } from './tree'

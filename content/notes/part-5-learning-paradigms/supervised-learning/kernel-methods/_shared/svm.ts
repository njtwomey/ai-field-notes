import { supportVectorMachine, type SupportVectorMachineModel } from 'aifn-methods/learning/kernel-methods'
import { rbf, linearKernel as aifnLinearKernel } from 'aifn-compute/learning/kernels'
import { dataset } from 'aifn-compute/learning/estimators'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'

export type Point = [number, number]
export type Kernel2 = (a: Point, b: Point) => number

export const linearKernel: Kernel2 = (a, b) => a[0] * b[0] + a[1] * b[1]
export const rbfKernel =
  (ell: number): Kernel2 =>
  (a, b) =>
    Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ell * ell))

export type SvmFit = {
  /** Dual variables, one per training point, in [0, C]. */
  alpha: number[]
  b: number
  iterations: number
  /** Largest KKT violation m(α) − M(α) at termination. */
  gap: number
  _model?: SupportVectorMachineModel
}

/**
 * Solves min ½ αᵀQα − 1ᵀα subject to 0 ≤ α ≤ C and yᵀα = 0, backed by aifn-methods.
 */
export function trainSvm(x: Point[], y: number[], C: number, kernel: Kernel2, tol = 1e-4, maxIter = 20000): SvmFit {
  const n = x.length
  const flat = Float64Array.from(x.flat())
  const X = fromData(flat, [n, 2])
  const y01 = fromData(
    Float64Array.from(y, (v) => (v > 0 ? 1 : 0)),
    [n],
  )
  // Detect if RBF or linear
  const isLinear = kernel === linearKernel
  const aifnKernel = isLinear ? aifnLinearKernel() : rbf({ lengthscale: 1 })
  const model = supportVectorMachine({
    C,
    kernel: aifnKernel,
    tolerance: tol,
    maxSteps: maxIter,
  }).fit(dataset(X, y01))
  return {
    alpha: Array.from(toFlat(model.alpha)),
    b: model.bias,
    iterations: model.steps,
    gap: model.gap,
    _model: model,
  }
}

/** f(x) = Σ αᵢ yᵢ k(xᵢ, x) + b, backed by aifn-methods model.score when available. */
export function decision(fit: SvmFit, x: Point[], y: number[], kernel: Kernel2, at: Point): number {
  if (fit._model) {
    const atTensor = fromData(Float64Array.from(at), [1, 2])
    return toFlat(fit._model.score(atTensor))[0]
  }
  let s = fit.b
  for (let i = 0; i < x.length; i++) if (fit.alpha[i] > 0) s += fit.alpha[i] * y[i] * kernel(x[i], at)
  return s
}

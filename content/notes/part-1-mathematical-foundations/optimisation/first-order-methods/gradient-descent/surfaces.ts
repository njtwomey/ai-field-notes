/**
 * The two problems of this note's figures, built on the engine: seeded data, the loss as an engine objective, its
 * log₁₀ surface over a (w, b) grid, and the exact optimum from the engine's estimators.
 *
 * - Linear regression with an uncentred feature: the MSE is a quadratic in (w, b) whose valley is long and diagonal.
 * - 1-D logistic regression with an uncentred feature: the cross-entropy valley is longer still, and not quadratic.
 */
import { hessian } from 'aifn-compute/foundation/autodiff'
import type { Objective } from 'aifn-compute/foundation/contracts'
import { child, integers, normal, stream, uniform } from 'aifn-compute/foundation/random'
import {
  add,
  dense,
  linspace,
  matmul,
  mean,
  mul,
  ones,
  reshape,
  stack,
  tensor,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { binaryCrossEntropyWithLogits, meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import { eigh } from 'aifn-compute/numerics/linalg'
import { objectiveFn, type ObjectiveFn } from 'aifn-compute/optim'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { linearRegression } from 'aifn-methods/learning/linear'

export type Vec = [number, number]

/** A loss over (w, b) with the data it was computed on and its log₁₀ surface on a grid. */
export type Surface = {
  /** The data: x and y (y is 0/1 for the logistic problem). */
  x: number[]
  y: number[]
  /** Grid axes and log₁₀ loss, `z[i][j]` at (w[j], b[i]). */
  w: number[]
  b: number[]
  z: number[][]
  /** The exact minimiser (w*, b*) and the loss there. */
  optimum: Vec
  optimumLoss: number
  /** The loss and its gradient at a point, for the optimisers. */
  f: ObjectiveFn
}

const values = (t: Tensor) => Array.from(dense.data(t))

/** The design [x, 1], n × 2, so that the predictions at θ = (w, b) are design · θ. */
const designOf = (x: Tensor) => stack([x, ones([x.shape[0]])], 1)

/** A loss of the predictions `design · θ`, as an engine objective of θ = (w, b). */
function objective(name: string, design: Tensor, loss: (predictions: Value) => Value): Objective {
  return { kind: 'objective', name, dim: 2, value: (theta) => loss(matmul(design, theta)) }
}

/**
 * log₁₀ of a loss at every (w[j], b[i]) in one batch: the predictions for all grid points are one n × G product, the
 * loss is taken per entry and averaged over the n examples.
 */
function logGrid(design: Tensor, w: number[], b: number[], perEntry: (predictions: Tensor) => Tensor): number[][] {
  const thetas = tensor([b.flatMap(() => w), b.flatMap((bi) => w.map(() => bi))])
  const losses = values(mean(perEntry(matmul(design, thetas)), 0) as Tensor)
  return b.map((_, i) => losses.slice(i * w.length, (i + 1) * w.length).map((l) => Math.log10(l)))
}

const axis = (lo: number, hi: number, n: number) => values(linspace(lo, hi, n)).map((v) => Math.round(v * 1e4) / 1e4)

/** y = 1.5x − 0.5 + noise with x uniform on [0, 4] and noise of standard deviation 0.5; 50 points, seed 0. */
export function regressionSurface(n = 50, seed = 0): Surface {
  const s = stream(seed)
  const x = uniform(child(s, 'x'), 0, 4, { shape: [n] })
  const y = add(add(mul(x, 1.5), -0.5), normal(child(s, 'noise'), 0, 0.5, { shape: [n] }))
  const design = designOf(x)
  const mse = objective('mse', design, (p) => meanSquaredErrorLoss(p, y))
  const f = objectiveFn(mse)
  const fit = linearRegression().fit(dataset(reshape(x, [n, 1]), y))
  const optimum: Vec = [dense.data(fit.weights)[0], fit.intercept]
  const w = axis(-1.5, 4.5, 61)
  const b = axis(-5, 4, 61)
  const yColumn = reshape(y, [n, 1])
  return {
    x: values(x),
    y: values(y),
    w,
    b,
    z: logGrid(design, w, b, (p) => meanSquaredErrorLoss(p, yColumn, { reduction: 'none' }) as Tensor),
    optimum,
    optimumLoss: f(tensor(optimum)).value,
    f,
  }
}

/** The eigenvalues (λ_min, λ_max) of the Hessian of a quadratic objective, which is the same at every point. */
export function curvature(surface: Surface): { lambdaMin: number; lambdaMax: number } {
  const design = designOf(tensor(surface.x))
  const y = tensor(surface.y)
  const H = hessian((theta: Tensor) => meanSquaredErrorLoss(matmul(design, theta), y))(tensor([0, 0])) as Tensor
  const [lambdaMax, lambdaMin] = values(eigh(H).values)
  return { lambdaMin, lambdaMax }
}

/** Two unit-variance classes centred at x = 3 (y = 0) and x = 5 (y = 1), labels equally likely; 100 points, seed 0. */
export function logisticSurface(n = 100, seed = 0): Surface {
  const s = stream(seed)
  const y = tensor(dense.data(integers(child(s, 'y'), 2, { shape: [n] })))
  const x = normal(child(s, 'x'), add(mul(y, 2), 3), 1)
  const design = designOf(x)
  const crossEntropy = objective('cross-entropy', design, (z) => binaryCrossEntropyWithLogits(z, y))
  const f = objectiveFn(crossEntropy)
  const fit = logisticRegression({ l2: 0 }).fit(dataset(reshape(x, [n, 1]), y))
  const optimum: Vec = [dense.data(fit.weights)[0], dense.data(fit.intercept)[0]]
  const w = axis(-1, 5, 81)
  const b = axis(-20, 4, 81)
  const yColumn = reshape(y, [n, 1])
  return {
    x: values(x),
    y: values(y),
    w,
    b,
    z: logGrid(design, w, b, (z) => binaryCrossEntropyWithLogits(z, yColumn, { reduction: 'none' }) as Tensor),
    optimum,
    optimumLoss: f(tensor(optimum)).value,
    f,
  }
}

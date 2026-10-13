import { stream } from 'aifn-compute/foundation/random'
import { matmul, mean, reshape, type Tensor, tensor, toFlat, toRows, transpose } from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { blobs, flippedMask, gaussians, withLabelNoise } from 'aifn-methods/data/synthetic'

/** Two features and a 0/1 label per example, as plain arrays for the charts. */
export type Points = { x: number[]; y: number[]; group: number[] }

/** Mean cross-entropy (bias fixed at 0) on the square grid `axis` × `axis`: `z[i][j]` is the loss at w = (axis[j], axis[i]). */
export type Surface = { axis: number[]; z: number[][] }

type Dataset = ReturnType<typeof blobs>

const N = 200

/** Two unit-variance Gaussian classes whose means are `separation` apart along the diagonal (1, 1). */
function diagonalBlobs(key: string, separation: number): Dataset {
  const h = separation / (2 * Math.SQRT2)
  return blobs(stream(key), {
    n: N,
    centers: [
      [-h, -h],
      [h, h],
    ],
  })
}

/** Features with correlation `rho`; the class means differ by `gap` along the low-variance axis (1, −1)/√2. */
function correlated(key: string, rho: number, gap: number): Dataset {
  const c = gap / (2 * Math.SQRT2)
  const cov = [
    [1, rho],
    [rho, 1],
  ]
  return gaussians(stream(key), {
    n: N,
    means: [
      [-c, c],
      [c, -c],
    ],
    covariances: [cov, cov],
  })
}

/** The labels of a labelled generator's dataset. */
function labels(d: Dataset): Tensor {
  if (!d.y) throw new Error('logistic-regression: dataset without labels')
  return d.y
}

function points(d: Dataset): Points {
  const rows = toRows(d.x)
  return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), group: toFlat(labels(d)) }
}

/** The loss surface of `d` over [−half, half]² on a `size` × `size` grid, from the engine's cross-entropy with logits. */
function surface(d: Dataset, half: number, size = 81): Surface {
  const axis = Array.from({ length: size }, (_, j) => -half + (2 * half * j) / (size - 1))
  const grid = new Float64Array(2 * size * size)
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      grid[2 * (i * size + j)] = axis[j]
      grid[2 * (i * size + j) + 1] = axis[i]
    }
  // Logits of every example at every grid point: (size² × 2)(2 × n).
  const logits = matmul(tensor(grid, [size * size, 2]), transpose(d.x))
  const targets = reshape(labels(d), [1, N])
  const losses = binaryCrossEntropyWithLogits(logits, targets, { reduction: 'none' }) as Tensor
  return { axis, z: toRows(reshape(mean(losses, 1), [size, size])) }
}

/** Blobs 3 apart: the data of the loss-surface explorer and the overlapping dataset of the solver race. */
const overlapping = () => diagonalBlobs('logistic-regression/overlapping', 3)

/** The data and loss surface of the loss-surface explorer: w over [−4, 4]². */
export function explorerData(): { data: Points; surface: Surface } {
  const d = overlapping()
  return { data: points(d), surface: surface(d, 4) }
}

export const SOLVER_DATASETS = ['overlapping', 'correlated', 'noisy labels', 'separable'] as const
export type SolverDatasetName = (typeof SOLVER_DATASETS)[number]

export type SolverDataset = {
  description: string
  /** True when the line x₁ + x₂ = 0 separates the classes, so the unpenalised loss has no minimum. */
  separable: boolean
  data: Points
  surface: Surface
}

function describe(name: SolverDatasetName, d: Dataset): string {
  switch (name) {
    case 'overlapping':
      return 'Two overlapping Gaussian classes. A well-conditioned bowl.'
    case 'correlated':
      return 'Features with correlation 0.95; the classes differ along the narrow axis. An ill-conditioned valley.'
    case 'noisy labels': {
      const flipped = flippedMask(d).reduce((a, v) => a + v, 0)
      return `Overlapping classes; each label flipped with probability 0.15 (${flipped} of ${N} flipped). A shallow bowl.`
    }
    case 'separable':
      return 'A line through the origin separates the classes. The loss has no minimum.'
  }
}

function solverDataset(name: SolverDatasetName): Dataset {
  switch (name) {
    case 'overlapping':
      return overlapping()
    case 'correlated':
      return correlated('logistic-regression/solver/correlated', 0.95, 0.25)
    case 'noisy labels':
      // withLabelNoise keeps the labels but is typed with the unlabelled Dataset (engine v0.2.9).
      return withLabelNoise(
        stream('logistic-regression/solver/noisy/flips'),
        diagonalBlobs('logistic-regression/solver/noisy', 2.5),
        { rate: 0.15 },
      ) as Dataset
    case 'separable':
      return diagonalBlobs('logistic-regression/solver/separable', 6)
  }
}

/** One dataset of the solver race with its unpenalised loss surface over [−6, 6]². */
export function solverData(name: SolverDatasetName): SolverDataset {
  const d = solverDataset(name)
  const data = points(d)
  const separable = data.group.every((g, i) => (2 * g - 1) * (data.x[i] + data.y[i]) > 0)
  return { description: describe(name, d), separable, data, surface: surface(d, 6) }
}

/**
 * Shared figure helpers for the low-density separation notes: the data sets (two moons, two Gaussian clusters),
 * labelled points chosen by handle, a random Fourier feature map from the engine, a prediction grid and a fixed-height
 * panel wrapper.
 */
import { useContext, type ReactNode } from 'react'
import { FrameContext } from 'aifn-render'
import { blobs, moons } from 'aifn-methods/data/synthetic'
import { randomFourierFeatures } from 'aifn-methods/learning/preprocessing'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, linspace, tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** A two-class data set as plain rows and classes. */
export interface PointData {
  rows: number[][]
  y: number[]
}

/** Two moons, shifted so that the pair is centred on the origin (inputs in about [−1.5, 1.5] × [−0.75, 0.75]). */
export function centredMoons(seed: number, noise: number, n = 200): PointData {
  const d = moons(stream(seed), { n, noise, spacing: 'random' })
  return { rows: (toArray(d.x) as number[][]).map(([a, b]) => [a - 0.5, b - 0.25]), y: toFlat(d.y) }
}

/** Two isotropic Gaussian clusters of equal size at (∓1.1, 0) with standard deviation `sd`. */
export function twoClusters(seed: number, sd: number, n = 200): PointData {
  const d = blobs(stream(seed), {
    n: [n / 2, n / 2],
    centers: [
      [-1.1, 0],
      [1.1, 0],
    ],
    sd,
  })
  return { rows: toArray(d.x) as number[][], y: toFlat(d.y) }
}

/** The index of the point of class `c` nearest to `anchor` (−1 if the class is empty). */
export function nearestOfClass(data: PointData, c: number, anchor: readonly [number, number]): number {
  let best = -1
  let bestD = Infinity
  data.rows.forEach((r, i) => {
    if (data.y[i] !== c) return
    const d = (r[0] - anchor[0]) ** 2 + (r[1] - anchor[1]) ** 2
    if (d < bestD) {
      bestD = d
      best = i
    }
  })
  return best
}

/**
 * The labelled points chosen by anchors: for each class, the points of that class nearest to its anchors (duplicates
 * removed). Returns the indices, their classes, the point each anchor snapped to (`byAnchor[c][a]`), and the indices
 * of the remaining (unlabelled) points.
 */
export function labelledByAnchors(data: PointData, anchors: readonly (readonly (readonly [number, number])[])[]) {
  const labelled: number[] = []
  const classes: number[] = []
  const byAnchor = anchors.map((list, c) =>
    list.map((a) => {
      const i = nearestOfClass(data, c, a)
      if (i >= 0 && !labelled.includes(i)) {
        labelled.push(i)
        classes.push(c)
      }
      return i
    }),
  )
  const unlabelled = data.rows.map((_, i) => i).filter((i) => !labelled.includes(i))
  return { labelled, classes, byAnchor, unlabelled }
}

/**
 * A fixed random Fourier feature map of two-dimensional inputs (the engine's `randomFourierFeatures`): `components`
 * features approximating a squared-exponential kernel of the given lengthscale, drawn once from `seed`.
 */
export function randomFeatureMap(components: number, lengthscale: number, seed: number) {
  const features = randomFourierFeatures({ components, lengthscale }).fit(
    { x: tensor([[0, 0]]) },
    { stream: stream(seed) },
  )
  return (rows: readonly (readonly number[])[]): number[][] =>
    rows.length === 0 ? [] : (toArray(features.transform(fromRows(rows))) as number[][])
}

/** A rectangular grid of prediction points: the axis values and the points row by row (y outer, x inner). */
export function predictionGrid(
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  nx: number,
  ny: number,
): { gx: number[]; gy: number[]; rows: number[][] } {
  const gx = toFlat(linspace(xRange[0], xRange[1], nx))
  const gy = toFlat(linspace(yRange[0], yRange[1], ny))
  return { gx, gy, rows: gy.flatMap((y) => gx.map((x) => [x, y])) }
}

/** Values listed row by row over an nx × ny grid, as Raster rows, through an optional transform. */
export function gridField(
  values: ArrayLike<number>,
  nx: number,
  ny: number,
  transform: (v: number) => number = (v) => v,
): number[][] {
  return Array.from({ length: ny }, (_, r) => Array.from({ length: nx }, (_, c) => transform(values[r * nx + c])))
}

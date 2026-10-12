/**
 * The data of the model-selection figure, generated once on the aifn-engine: 400 points from three axis-aligned
 * Gaussians. Two are long and thin, parallel and close together; the third is small and round.
 */
import { gaussians, shuffleDataset } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import type { Vec2 as Point } from 'aifn-compute/numerics/linalg'

const diagonal = (sx: number, sy: number) => [
  [sx * sx, 0],
  [0, sy * sy],
]

function generate() {
  const s = stream('gaussian-mixture-model/blobs')
  const data = gaussians(s, {
    n: 400,
    classWeights: [0.4, 0.4, 0.2],
    means: [
      [0, 1.2],
      [0, -1.2],
      [4, 3.5],
    ],
    covariances: [diagonal(2.2, 0.35), diagonal(2.2, 0.35), diagonal(0.4, 0.4)],
  })
  // A random order, so that the first n points are a random subsample of all 400.
  const points = toRows(shuffleDataset(child(s, 'order'), data).x) as Point[]
  return { points, x: points.map((p) => p[0]), y: points.map((p) => p[1]) }
}

let cached: ReturnType<typeof generate> | undefined

/** The points as rows, and their coordinates as columns for plotting. Generated on first use. */
export const blobData = () => (cached ??= generate())

/** The data of both k-means figures: 300 points in three round Gaussian blobs, generated once on the aifn-engine. */
import { blobs as makeBlobs, shuffleDataset } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import type { Vec2 as Point } from 'aifn-compute/numerics/linalg'

/** Blob centres on a circle of radius 3, 120° apart. */
const CENTRES = [0, 1, 2].map((j): Point => [3 * Math.cos((2 * Math.PI * j) / 3), 3 * Math.sin((2 * Math.PI * j) / 3)])

function generate() {
  const s = stream('k-means/blobs')
  // 100 points per blob with standard deviation 0.8, in a random order.
  const data = shuffleDataset(child(s, 'order'), makeBlobs(s, { n: 300, centers: CENTRES, sd: 0.8 }))
  const points = toRows(data.x) as Point[]
  return { points, x: points.map((p) => p[0]), y: points.map((p) => p[1]) }
}

let cached: ReturnType<typeof generate> | undefined

/** The points as rows, and their coordinates as columns for plotting. Generated on first use. */
export const blobData = () => (cached ??= generate())

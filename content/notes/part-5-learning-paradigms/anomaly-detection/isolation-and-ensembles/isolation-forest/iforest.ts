import { averagePathLength } from 'aifn-methods/unsupervised/anomaly'
import { stream, uniform as drawUniform } from 'aifn-compute/foundation/random'

export type Pt = [number, number]
export type Box = { x0: number; x1: number; y0: number; y1: number }
export type Segment = { from: [number, number]; to: [number, number] }

type Node = { size: number } | { dim: 0 | 1; split: number; left: Node; right: Node }

/** Average leaf depth of a random binary tree with n leaves, delegated to aifn-methods. */
export const avgPathLength = averagePathLength

/**
 * Grow one isolation tree on `idx`. A split picks a coordinate at random and a value uniformly between the node's
 * minimum and maximum on it. When `cuts` is given, each split is recorded as a segment across the node's region.
 */
function grow(
  pts: Pt[],
  idx: number[],
  depth: number,
  limit: number,
  uniform: () => number,
  region: Box,
  cuts?: Segment[],
): Node {
  if (idx.length <= 1 || depth >= limit) return { size: idx.length }
  const lo = [Infinity, Infinity]
  const hi = [-Infinity, -Infinity]
  for (const i of idx) {
    for (const d of [0, 1]) {
      lo[d] = Math.min(lo[d], pts[i][d])
      hi[d] = Math.max(hi[d], pts[i][d])
    }
  }
  const dims = ([0, 1] as const).filter((d) => hi[d] > lo[d])
  if (dims.length === 0) return { size: idx.length }
  const dim = dims[Math.floor(uniform() * dims.length)]
  const split = lo[dim] + uniform() * (hi[dim] - lo[dim])
  const left = idx.filter((i) => pts[i][dim] < split)
  const right = idx.filter((i) => pts[i][dim] >= split)
  let leftRegion = region
  let rightRegion = region
  if (dim === 0) {
    cuts?.push({ from: [split, region.y0], to: [split, region.y1] })
    leftRegion = { ...region, x1: split }
    rightRegion = { ...region, x0: split }
  } else {
    cuts?.push({ from: [region.x0, split], to: [region.x1, split] })
    leftRegion = { ...region, y1: split }
    rightRegion = { ...region, y0: split }
  }
  return {
    dim,
    split,
    left: grow(pts, left, depth + 1, limit, uniform, leftRegion, cuts),
    right: grow(pts, right, depth + 1, limit, uniform, rightRegion, cuts),
  }
}

/** Path length h(x): edges to the leaf, plus c(size) for the points a depth-limited leaf still holds. */
function pathLength(node: Node, x: Pt, depth = 0): number {
  if ('size' in node) return depth + avgPathLength(node.size)
  return pathLength(x[node.dim] < node.split ? node.left : node.right, x, depth + 1)
}

export type Forest = { trees: Node[]; psi: number; firstSample: number[]; firstCuts: Segment[] }

export function buildForest(pts: Pt[], nTrees: number, psi: number, seed: number, region: Box): Forest {
  const s = stream(`iforest/${seed}`)
  const uniform = () => drawUniform(s)
  const size = Math.min(psi, pts.length)
  const limit = Math.ceil(Math.log2(Math.max(size, 2)))
  const trees: Node[] = []
  let firstSample: number[] = []
  const firstCuts: Segment[] = []
  for (let t = 0; t < nTrees; t++) {
    // Partial Fisher–Yates shuffle: a subsample of `size` indices without replacement.
    const all = pts.map((_, i) => i)
    for (let i = 0; i < size; i++) {
      const j = i + Math.floor(uniform() * (all.length - i))
      ;[all[i], all[j]] = [all[j], all[i]]
    }
    const sample = all.slice(0, size)
    if (t === 0) firstSample = sample
    trees.push(grow(pts, sample, 0, limit, uniform, region, t === 0 ? firstCuts : undefined))
  }
  return { trees, psi: size, firstSample, firstCuts }
}

/** Mean path length over the trees and the anomaly score s = 2^(−E[h] / c(ψ)). */
export function isolationScore(f: Forest, x: Pt): { meanPath: number; score: number } {
  const meanPath = f.trees.reduce((s, t) => s + pathLength(t, x), 0) / f.trees.length
  return { meanPath, score: Math.pow(2, -meanPath / avgPathLength(f.psi)) }
}

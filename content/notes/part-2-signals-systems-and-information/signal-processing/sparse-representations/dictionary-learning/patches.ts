import { dctMatrix } from 'aifn-compute/foundation/fourier'
import { permutation, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** The patch side in pixels; a patch is a vector of P² values. */
export const P = 6
/** The image side in pixels: 10 × 10 tiles of P × P. */
export const W = 60

/** 1 inside the shape, 0 outside, blended over about one pixel of signed distance d (negative inside). */
const inside = (d: number) => Math.min(1, Math.max(0, 0.5 - d))

/** The scene at a point (u, v) in pixels, v downwards: a disc, a triangle, a tilted square and a patch of stripes. */
function scene(u: number, v: number): number {
  const disc = inside(Math.hypot(u - 16, v - 16) - 11)
  // Triangle (35, 5), (57, 5), (46, 27): the largest signed distance to its three edges.
  const edges = [
    [0, -1, 5],
    [22 / Math.hypot(22, 11), 11 / Math.hypot(22, 11), (22 * 57 + 11 * 5) / Math.hypot(22, 11)],
    [-22 / Math.hypot(22, 11), 11 / Math.hypot(22, 11), (-22 * 35 + 11 * 5) / Math.hypot(22, 11)],
  ]
  const tri = inside(Math.max(...edges.map(([a, b, c]) => a * u + b * v - c)))
  const c = Math.cos(0.35)
  const s = Math.sin(0.35)
  const du = u - 45
  const dv = v - 45
  const square = inside(Math.max(Math.abs(c * du + s * dv), Math.abs(-s * du + c * dv)) - 9)
  const inStripes = inside(Math.max(u - 28, 32 - v, 2 - u, v - 58))
  const stripes = 0.5 + 0.5 * Math.sin((2 * Math.PI * (u * Math.cos(0.6) + v * Math.sin(0.6))) / 7)
  return Math.max(disc, 0.7 * tri, 0.85 * square, inStripes * stripes)
}

/** The image, W × W, row-major from the top row, each pixel the mean of 3 × 3 samples of the scene. */
export function makeImage(): Float64Array {
  const img = new Float64Array(W * W)
  for (let r = 0; r < W; r++)
    for (let c = 0; c < W; c++) {
      let sum = 0
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) sum += scene(c + (b + 0.5) / 3, r + (a + 0.5) / 3)
      img[r * W + c] = sum / 9
    }
  return img
}

/** Patches with their means removed, as the columns of a P² × n matrix, and the means. */
export type Patches = { Y: number[][]; means: number[] }

/** The P × P patches with top-left corners at the given (row, column) positions, mean removed. */
function patchesAt(img: Float64Array, at: readonly (readonly [number, number])[]): Patches {
  const Y = Array.from({ length: P * P }, () => new Array<number>(at.length).fill(0))
  const means = at.map(([r0, c0], i) => {
    let mean = 0
    for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) mean += img[(r0 + r) * W + c0 + c]
    mean /= P * P
    for (let r = 0; r < P; r++) for (let c = 0; c < P; c++) Y[r * P + c][i] = img[(r0 + r) * W + c0 + c] - mean
    return mean
  })
  return { Y, means }
}

/**
 * n training patches at distinct positions drawn at random from the image, skipping flat ones (pixel standard deviation
 * below 0.02): with the mean removed they are zero and would give zero atoms to start from.
 */
export function trainingPatches(img: Float64Array, n: number, s: Stream): Patches {
  const side = W - P + 1
  const at: (readonly [number, number])[] = []
  for (const p of toFlat(permutation(s, side * side))) {
    const corner = [Math.floor(p / side), p % side] as const
    const { Y } = patchesAt(img, [corner])
    if (Y.reduce((sum, [v]) => sum + v * v, 0) / (P * P) >= 0.02 ** 2) at.push(corner)
    if (at.length === n) break
  }
  return patchesAt(img, at)
}

/** The (W / P)² non-overlapping tiles of the image, row by row. */
export function tiles(img: Float64Array): Patches {
  const t = W / P
  return patchesAt(
    img,
    Array.from({ length: t * t }, (_, i) => [Math.floor(i / t) * P, (i % t) * P] as const),
  )
}

/** The image rebuilt from tile codes: tile i is column i of D X plus its mean. */
export function fromTiles(D: number[][], X: number[][], means: readonly number[]): Float64Array {
  const t = W / P
  const img = new Float64Array(W * W)
  const k = X.length
  means.forEach((mean, i) => {
    const r0 = Math.floor(i / t) * P
    const c0 = (i % t) * P
    for (let p = 0; p < P * P; p++) {
      let v = mean
      for (let j = 0; j < k; j++) if (X[j][i] !== 0) v += D[p][j] * X[j][i]
      img[(r0 + Math.floor(p / P)) * W + c0 + (p % P)] = v
    }
  })
  return img
}

/**
 * The separable 2-D DCT atoms of a P × P patch except the constant one, as the columns of a P² × (P² − 1) matrix,
 * ordered by total frequency.
 */
export function dctDictionary(): number[][] {
  const C = toFlat(dctMatrix(P))
  const pairs: [number, number][] = []
  for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) if (a + b > 0) pairs.push([a, b])
  pairs.sort((x, y) => x[0] + x[1] - (y[0] + y[1]) || x[0] - y[0])
  return Array.from({ length: P * P }, (_, p) =>
    pairs.map(([a, b]) => C[a * P + Math.floor(p / P)] * C[b * P + (p % P)]),
  )
}

/** Tiles per row of an atom mosaic with k atoms. */
export const mosaicColumns = (k: number) => Math.ceil(Math.sqrt(k * 1.5))

/**
 * Atoms (the columns of D, P² × k) laid out as one grid for a raster: P × P tiles, each scaled to a largest magnitude
 * of 1, with a one-pixel gap (NaN) between them; rows listed from the top, for an inverted y axis.
 */
export function mosaic(D: number[][], order: readonly number[]): { z: number[][]; cols: number; rows: number } {
  const cols = mosaicColumns(order.length)
  const rows = Math.ceil(order.length / cols)
  const h = rows * (P + 1) - 1
  const w = cols * (P + 1) - 1
  const grid = Array.from({ length: h }, () => new Array<number>(w).fill(NaN))
  order.forEach((j, n) => {
    const r0 = Math.floor(n / cols) * (P + 1)
    const c0 = (n % cols) * (P + 1)
    // An atom and its negative are the same atom: divide by the entry of largest magnitude, so that entry shows as +1.
    let peak = 0
    for (let p = 0; p < P * P; p++) if (Math.abs(D[p][j]) > Math.abs(peak)) peak = D[p][j]
    for (let p = 0; p < P * P; p++) grid[r0 + Math.floor(p / P)][c0 + (p % P)] = peak !== 0 ? D[p][j] / peak : 0
  })
  return { z: grid, cols, rows }
}

/** An image (row-major from the top) as raster rows, top row first. */
export function imageRows(img: Float64Array): number[][] {
  return Array.from({ length: W }, (_, r) => Array.from(img.subarray(r * W, (r + 1) * W)))
}

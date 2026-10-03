/**
 * A patch memory bank on toy texture images, in the manner of PatchCore (Roth et al. 2022). Features are raw p × p
 * pixel neighbourhoods rather than pretrained network activations; the bank, the nearest-neighbour score, the maximum
 * over patches and the greedy coreset are the same.
 */
import type { Rand } from './deepOneClass'

export const SIZE = 28
export const PATCH = 5
const HALF = (PATCH - 1) / 2
/** Patch centres along each axis: HALF … SIZE − 1 − HALF. */
export const CENTRES = Array.from({ length: SIZE - 2 * HALF }, (_, i) => i + HALF)

export type Img = Float64Array // SIZE × SIZE, row-major
export type Spot = [number, number] // (column, row) in pixels

const ANGLE = Math.PI / 6
const PERIOD = 6

const bump = (col: number, row: number, at: Spot) => 0.8 * Math.exp(-((col - at[0]) ** 2 + (row - at[1]) ** 2) / 2)

/** Diagonal stripes of random phase with pixel noise, a bright rivet (normal) and optionally a dark dent (defect). */
export function texture(phase: number, rivet: Spot, dent: Spot | null, r: Rand): Img {
  const img = new Float64Array(SIZE * SIZE)
  for (let row = 0; row < SIZE; row++)
    for (let col = 0; col < SIZE; col++) {
      const t = (col * Math.cos(ANGLE) + row * Math.sin(ANGLE)) / PERIOD
      let v = 0.5 + 0.35 * Math.sin(2 * Math.PI * t + phase) + 0.04 * r.normal() + bump(col, row, rivet)
      if (dent) v -= bump(col, row, dent)
      img[row * SIZE + col] = v
    }
  return img
}

/** Every PATCH × PATCH neighbourhood, in the order of CENTRES (rows outer). */
export function patches(img: Img): Float64Array[] {
  const out: Float64Array[] = []
  for (const row of CENTRES)
    for (const col of CENTRES) {
      const f = new Float64Array(PATCH * PATCH)
      let k = 0
      for (let dr = -HALF; dr <= HALF; dr++)
        for (let dc = -HALF; dc <= HALF; dc++) f[k++] = img[(row + dr) * SIZE + col + dc]
      out.push(f)
    }
  return out
}

const dist = (a: Float64Array, b: Float64Array) => {
  let s = 0
  for (let k = 0; k < a.length; k++) s += (a[k] - b[k]) ** 2
  return Math.sqrt(s)
}

/**
 * Greedy k-centre coreset: start from the first patch and repeatedly add the patch farthest from those chosen. It
 * minimises, approximately, the largest distance from any bank patch to the coreset.
 */
export function greedyCoreset(bank: Float64Array[], k: number): number[] {
  const chosen = [0]
  const d = bank.map((p) => dist(p, bank[0]))
  while (chosen.length < k) {
    let best = 0
    for (let i = 1; i < d.length; i++) if (d[i] > d[best]) best = i
    chosen.push(best)
    for (let i = 0; i < d.length; i++) d[i] = Math.min(d[i], dist(bank[i], bank[best]))
  }
  return chosen
}

/** k indices drawn without replacement. */
export function randomSubset(n: number, k: number, r: Rand): number[] {
  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(r.uniform() * (n - i))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx.slice(0, k)
}

/** Distance from each test patch to its nearest memory patch, as a grid over CENTRES (rows outer). */
export function anomalyMap(img: Img, memory: Float64Array[]): number[][] {
  const q = patches(img)
  const n = CENTRES.length
  return CENTRES.map((_, i) =>
    CENTRES.map((_, j) => {
      const p = q[i * n + j]
      let best = Infinity
      for (const m of memory) best = Math.min(best, dist(p, m))
      return best
    }),
  )
}

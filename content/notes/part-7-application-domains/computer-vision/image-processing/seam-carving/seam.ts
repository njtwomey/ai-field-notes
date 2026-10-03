/**
 * Seam carving on small colour images. An image is row-major RGB in [0, 1]: channel k of pixel (r, c) is
 * `rgb[3 * (r * width + c) + k]`. Every image here has H rows; widths vary as seams are removed or inserted.
 */
import { rng } from '@/lib/math'

export const H = 96
export const W = 128
/** Seams precomputed per image: half the width can be removed, and a quarter inserted. */
export const MAX_REMOVE = 64
export const MAX_INSERT = 32

export type Energy = 'backward' | 'forward'
export type ImageId = 'landscape' | 'shapes' | 'checkerboard' | 'text' | 'line'
/** A rectangle in original pixel coordinates, rows r0…r1 and columns c0…c1 inclusive. */
export type Box = { r0: number; r1: number; c0: number; c1: number }
export type BoxMode = 'off' | 'protect' | 'remove'

/** Energy added to (protect) or subtracted from (remove) every pixel under the box. */
const MASK_WEIGHT = 10

// ---------------------------------------------------------------------------------------------------------------
// Test images

type Rgb3 = [number, number, number]
const mix = (a: Rgb3, b: Rgb3, t: number): Rgb3 => [0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * t) as Rgb3

function paint(f: (r: number, c: number) => Rgb3, noise = 0.008, seed = 3): Float32Array {
  const g = rng(seed)
  const out = new Float32Array(H * W * 3)
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const v = f(r, c)
      // A little seeded noise breaks exact ties between equal-cost seams in flat regions.
      for (let k = 0; k < 3; k++) out[3 * (r * W + c) + k] = Math.min(1, Math.max(0, v[k] + noise * g.normal()))
    }
  return out
}

const inDisk = (r: number, c: number, r0: number, c0: number, rad: number) => (r - r0) ** 2 + (c - c0) ** 2 <= rad ** 2

function landscape(): Float32Array {
  const horizon = 58
  return paint((r, c) => {
    const tree = (cc: number, top: number, rad: number, trunkTop: number, trunkBottom: number): Rgb3 | null => {
      if (inDisk(r, c, top, cc, rad)) return mix([0.16, 0.45, 0.18], [0.08, 0.3, 0.1], (r - top + rad) / (2 * rad))
      if (r >= trunkTop && r <= trunkBottom && Math.abs(c - cc) <= 1.5) return [0.42, 0.27, 0.14]
      return null
    }
    const t1 = tree(26, 42, 11, 48, 72)
    if (t1) return t1
    const t2 = tree(60, 50, 8, 54, 70)
    if (t2) return t2
    // A house: walls, a door and a pitched roof.
    if (r >= 62 && r <= 78 && c >= 86 && c <= 106)
      return c >= 94 && c <= 98 && r >= 70 ? [0.35, 0.2, 0.12] : [0.82, 0.36, 0.28]
    if (r >= 50 && r < 62 && Math.abs(c - 96) <= (r - 50) * 1.05) return [0.45, 0.2, 0.18]
    if (inDisk(r, c, 20, 112, 8)) return [1, 0.86, 0.32]
    if (r < horizon) return mix([0.36, 0.56, 0.86], [0.78, 0.87, 0.96], r / horizon)
    return mix([0.42, 0.66, 0.3], [0.26, 0.47, 0.2], (r - horizon) / (H - horizon))
  })
}

function shapes(): Float32Array {
  return paint((r, c) => {
    if (inDisk(r, c, 28, 26, 14)) return [0.86, 0.26, 0.24]
    if (r >= 50 && r <= 80 && c >= 50 && c <= 76) return [0.2, 0.4, 0.8]
    if (r >= 12 && r <= 44 && Math.abs(c - 100) <= (r - 12) * 0.5) return [0.95, 0.75, 0.15]
    if (inDisk(r, c, 76, 112, 9)) return [0.15, 0.65, 0.4]
    return mix([0.93, 0.9, 0.84], [0.8, 0.84, 0.9], (r + c) / (H + W))
  })
}

function checkerboard(): Float32Array {
  return paint((r, c) => ((Math.floor(r / 12) + Math.floor(c / 12)) % 2 ? [0.15, 0.2, 0.35] : [0.92, 0.9, 0.84]))
}

/** Lines of glyph-like marks: no column is free of strokes, so every seam cuts letters. */
function text(): Float32Array {
  const g = rng(11)
  const ink = new Uint8Array(H * W)
  for (let line = 0; line < 7; line++) {
    const top = 6 + line * 13
    let c = 4 + Math.floor(g.uniform() * 6)
    while (c < W - 8) {
      const letters = 2 + Math.floor(g.uniform() * 5)
      for (let l = 0; l < letters && c < W - 6; l++) {
        // A 5 × 8 glyph: one or two vertical strokes and up to three horizontal bars.
        const strokes = [g.uniform() < 0.8 ? 0 : -1, g.uniform() < 0.5 ? 4 : -1, g.uniform() < 0.2 ? 2 : -1]
        const bars = [g.uniform() < 0.5 ? 0 : -1, g.uniform() < 0.5 ? 4 : -1, g.uniform() < 0.5 ? 7 : -1]
        for (let dr = 0; dr < 8; dr++)
          for (let dc = 0; dc < 5; dc++) if (strokes.includes(dc) || bars.includes(dr)) ink[(top + dr) * W + c + dc] = 1
        c += 7
      }
      c += 5
    }
  }
  return paint((r, c) => (ink[r * W + c] ? [0.12, 0.12, 0.16] : [0.96, 0.94, 0.88]), 0.006, 5)
}

/** A straight dark line across a smooth background, with a patch of texture that seams avoid. */
function line(): Float32Array {
  const g = rng(17)
  const texture = Float32Array.from({ length: H * W }, () => g.uniform())
  return paint((r, c) => {
    // Distance from the line through (0, 18) and (95, 110).
    const d = Math.abs((110 - 18) * r - 95 * (c - 18)) / Math.hypot(92, 95)
    if (d <= 1.6) return [0.1, 0.1, 0.14]
    if (r >= 8 && r <= 50 && c >= 80 && c <= 122) {
      const t = texture[r * W + c]
      return [0.55 + 0.4 * t, 0.45 + 0.3 * t, 0.3]
    }
    return mix([0.9, 0.91, 0.94], [0.84, 0.88, 0.92], r / H)
  })
}

const MAKERS: Record<ImageId, () => Float32Array> = { landscape, shapes, checkerboard, text, line }
const images = new Map<ImageId, Float32Array>()
export function testImage(id: ImageId): Float32Array {
  let img = images.get(id)
  if (!img) images.set(id, (img = MAKERS[id]()))
  return img
}

// ---------------------------------------------------------------------------------------------------------------
// Costs and dynamic programming

/** L1 distance between two pixels over the three channels. */
function diff(rgb: Float32Array, a: number, b: number): number {
  return (
    Math.abs(rgb[3 * a] - rgb[3 * b]) +
    Math.abs(rgb[3 * a + 1] - rgb[3 * b + 1]) +
    Math.abs(rgb[3 * a + 2] - rgb[3 * b + 2])
  )
}

/**
 * The seam graph of an image of width w. `node[p]` is the cost of taking pixel p into a seam; `edge[3p + d + 1]` the
 * extra cost when p's parent in the row above is at column offset d ∈ {-1, 0, 1}. Backward energy puts the gradient
 * magnitude on the nodes and nothing on the edges. Forward energy puts only the mask on the nodes and puts on each
 * edge the colour differences of the pixels that become neighbours when the seam is removed.
 */
type Graph = { node: Float32Array; edge: Float32Array | null; shown: Float32Array }

function graph(rgb: Float32Array, mask: Float32Array, w: number, mode: Energy): Graph {
  const n = H * w
  const node = new Float32Array(n)
  const shown = new Float32Array(n)
  const edge = mode === 'forward' ? new Float32Array(3 * n) : null
  for (let r = 0; r < H; r++) {
    const up = Math.max(r - 1, 0)
    const down = Math.min(r + 1, H - 1)
    for (let c = 0; c < w; c++) {
      const p = r * w + c
      const left = r * w + Math.max(c - 1, 0)
      const right = r * w + Math.min(c + 1, w - 1)
      const m = MASK_WEIGHT * mask[p]
      if (!edge) {
        // |∂x I| + |∂y I| by central differences, summed over channels.
        const e = diff(rgb, right, left) / 2 + diff(rgb, down * w + c, up * w + c) / 2
        node[p] = e + m
        shown[p] = e + m
        continue
      }
      const cu = diff(rgb, right, left)
      node[p] = m
      shown[p] = cu + m
      if (r === 0) {
        edge.fill(cu, 3 * p, 3 * p + 3)
        continue
      }
      const above = up * w + c
      edge[3 * p] = cu + diff(rgb, above, left)
      edge[3 * p + 1] = cu
      edge[3 * p + 2] = cu + diff(rgb, above, right)
    }
  }
  return { node, edge, shown }
}

export type Fields = {
  /** M: cheapest cost of a seam from the top row down to each pixel. */
  M: Float32Array
  /** The chosen parent offset d ∈ {-1, 0, 1} of each pixel below the top row. */
  parent: Int8Array
  /** Cheapest cost from each pixel down to the bottom row, not counting the pixel itself. */
  below: Float32Array
  child: Int8Array
}

/**
 * M(r, c) = node(r, c) + min over d of [M(r-1, c+d) + edge(r, c, d)]. Offsets are tried in the order 0, -1, 1, so ties
 * go straight down, then left.
 */
function cumulative(g: Graph, w: number): Fields {
  const { node, edge } = g
  const n = H * w
  const M = new Float32Array(n)
  const parent = new Int8Array(n)
  for (let c = 0; c < w; c++) M[c] = node[c] + (edge ? edge[3 * c + 1] : 0)
  for (let r = 1; r < H; r++)
    for (let c = 0; c < w; c++) {
      const p = r * w + c
      let best = Infinity
      let bestD = 0
      for (const d of [0, -1, 1]) {
        const cc = c + d
        if (cc < 0 || cc >= w) continue
        const v = M[p - w + d] + (edge ? edge[3 * p + d + 1] : 0)
        if (v < best) {
          best = v
          bestD = d
        }
      }
      M[p] = node[p] + best
      parent[p] = bestD
    }
  // The same recursion run upwards, for the cheapest seam through a given pixel.
  const below = new Float32Array(n)
  const child = new Int8Array(n)
  for (let r = H - 2; r >= 0; r--)
    for (let c = 0; c < w; c++) {
      let best = Infinity
      let bestD = 0
      for (const d of [0, -1, 1]) {
        const cc = c + d
        if (cc < 0 || cc >= w) continue
        const q = (r + 1) * w + cc
        // Pixel q's parent sits at offset -d from it.
        const v = below[q] + node[q] + (edge ? edge[3 * q - d + 1] : 0)
        if (v < best) {
          best = v
          bestD = d
        }
      }
      below[r * w + c] = best
      child[r * w + c] = bestD
    }
  return { M, parent, below, child }
}

/** Backtrack from the smallest M in the bottom row. Returns the seam's column in each row, top to bottom. */
function minimumSeam(f: Fields, w: number): { seam: Int32Array; cost: number } {
  const seam = new Int32Array(H)
  let c = 0
  for (let j = 1; j < w; j++) if (f.M[(H - 1) * w + j] < f.M[(H - 1) * w + c]) c = j
  const cost = f.M[(H - 1) * w + c]
  for (let r = H - 1; r >= 0; r--) {
    seam[r] = c
    if (r > 0) c += f.parent[r * w + c]
  }
  return { seam, cost }
}

/** The cheapest seam that passes through pixel (r, c), and its cost. */
export function seamThrough(v: View, r: number, c: number): { seam: number[]; cost: number } {
  const { width: w, fields: f } = v
  const seam = new Array<number>(H)
  seam[r] = c
  for (let i = r, j = c; i > 0; i--) seam[i - 1] = j += f.parent[i * w + j]
  for (let i = r, j = c; i < H - 1; i++) seam[i + 1] = j += f.child[i * w + j]
  return { seam, cost: f.M[r * w + c] + f.below[r * w + c] }
}

// ---------------------------------------------------------------------------------------------------------------
// The removal sequence

export function boxMask(box: Box | null, mode: BoxMode): Float32Array {
  const mask = new Float32Array(H * W)
  if (!box || mode === 'off') return mask
  const s = mode === 'protect' ? 1 : -1
  for (let r = box.r0; r <= box.r1; r++) for (let c = box.c0; c <= box.c1; c++) mask[r * W + c] = s
  return mask
}

export type Sequence = {
  rgb: Float32Array
  mask: Float32Array
  mode: Energy
  /** The step (0-based) at which each original pixel is removed; MAX_REMOVE if it survives every step. */
  removedAt: Int16Array
  /** Display ranges fixed at step 0, so colours stay put as seams are removed. */
  shownRange: [number, number]
  costRange: [number, number]
}

function quantile(values: Float32Array, q: number): number {
  const sorted = Float32Array.from(values).sort()
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
}

/** Remove MAX_REMOVE seams one at a time, recording when each original pixel goes. */
export function carve(rgb: Float32Array, mask: Float32Array, mode: Energy): Sequence {
  const removedAt = new Int16Array(H * W).fill(MAX_REMOVE)
  let src = Int32Array.from({ length: H * W }, (_, i) => i)
  let cur = rgb
  let curMask = mask
  let w = W
  let shownRange: [number, number] = [0, 1]
  let costRange: [number, number] = [0, 1]
  for (let step = 0; step < MAX_REMOVE; step++) {
    const g = graph(cur, curMask, w, mode)
    const f = cumulative(g, w)
    if (step === 0) {
      const unmasked = mask.some((m) => m !== 0) ? graph(rgb, new Float32Array(H * W), W, mode).shown : g.shown
      const top = quantile(unmasked, 0.98)
      shownRange = mask.some((m) => m < 0) ? [-top, top] : [0, top]
      costRange = [Math.min(0, quantile(f.M, 0)), quantile(f.M, 0.99)]
    }
    const { seam } = minimumSeam(f, w)
    const nextW = w - 1
    const nextSrc = new Int32Array(H * nextW)
    const next = new Float32Array(H * nextW * 3)
    const nextMask = new Float32Array(H * nextW)
    for (let r = 0; r < H; r++) {
      removedAt[src[r * w + seam[r]]] = step
      for (let c = 0, o = r * nextW; c < w; c++) {
        if (c === seam[r]) continue
        const p = r * w + c
        nextSrc[o] = src[p]
        nextMask[o] = curMask[p]
        next.set(cur.subarray(3 * p, 3 * p + 3), 3 * o)
        o++
      }
    }
    src = nextSrc
    cur = next
    curMask = nextMask
    w = nextW
  }
  return { rgb, mask, mode, removedAt, shownRange, costRange }
}

export type View = {
  width: number
  rgb: Float32Array
  /** The original column each pixel came from; an inserted pixel reports the column it was copied from. */
  src: Int32Array
  shown: Float32Array
  fields: Fields
  /** The next seam to remove, and its cost. */
  seam: Int32Array
  cost: number
}

/**
 * The image at a target width. Narrower: drop the pixels removed in the first W - width steps. Wider: take the first
 * width - W seams of the removal sequence and insert beside each of their pixels a new pixel, the mean of it and its
 * right-hand neighbour in the original image.
 */
export function viewAt(s: Sequence, width: number): View {
  const k = width - W
  const rgb = new Float32Array(H * width * 3)
  const mask = new Float32Array(H * width)
  const src = new Int32Array(H * width)
  for (let r = 0; r < H; r++) {
    let o = r * width
    for (let c = 0; c < W; c++) {
      const p = r * W + c
      const t = s.removedAt[p]
      if (k < 0 && t < -k) continue
      src[o] = c
      mask[o] = s.mask[p]
      rgb.set(s.rgb.subarray(3 * p, 3 * p + 3), 3 * o)
      o++
      if (k > 0 && t < k) {
        const q = r * W + Math.min(c + 1, W - 1)
        for (let ch = 0; ch < 3; ch++) rgb[3 * o + ch] = (s.rgb[3 * p + ch] + s.rgb[3 * q + ch]) / 2
        src[o] = c
        mask[o] = s.mask[p]
        o++
      }
    }
  }
  const g = graph(rgb, mask, width, s.mode)
  const fields = cumulative(g, width)
  const { seam, cost } = minimumSeam(fields, width)
  return { width, rgb, src, shown: g.shown, fields, seam, cost }
}

/** Plain resampling of every row to `width` columns by linear interpolation, for comparison. */
export function scaled(rgb: Float32Array, width: number): Float32Array {
  const out = new Float32Array(H * width * 3)
  for (let c = 0; c < width; c++) {
    const x = width > 1 ? (c * (W - 1)) / (width - 1) : 0
    const c0 = Math.floor(x)
    const c1 = Math.min(c0 + 1, W - 1)
    const t = x - c0
    for (let r = 0; r < H; r++)
      for (let ch = 0; ch < 3; ch++)
        out[3 * (r * width + c) + ch] = (1 - t) * rgb[3 * (r * W + c0) + ch] + t * rgb[3 * (r * W + c1) + ch]
  }
  return out
}

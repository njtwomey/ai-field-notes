import { normal, stream } from 'aifn-compute/foundation/random'
/**
 * Small greyscale-image helpers for the image-processing notes. An image is row-major, `img[r][c]`, with values in
 * [0, 1]. Everything here is light enough to recompute on a 64 × 64 image during a slider drag.
 */

export type Image = number[][]
export type Kernel = number[][]
export type Border = 'zero' | 'replicate' | 'reflect'

export const SIZE = 64

/**
 * A 64 × 64 test image: a bright rectangle, a grey disk, a triangle and a checkerboard on a dark background, so that
 * straight edges, curved edges, corners and texture all appear. `noise` adds seeded Gaussian noise of that standard
 * deviation.
 */
export function testImage(noise = 0, seed = 7): Image {
  const g = stream(seed)
  return Array.from({ length: SIZE }, (_, r) =>
    Array.from({ length: SIZE }, (_, c) => {
      let v = 0.15
      if (r >= 8 && r <= 27 && c >= 6 && c <= 25) v = 0.9
      if ((r - 46) ** 2 + (c - 16) ** 2 <= 11 ** 2) v = 0.6
      // Triangle with apex at the top (high row index) and base on row 6.
      if (r >= 6 && r <= 28 && Math.abs(c - 48) <= (28 - r) * 0.55) v = 0.7
      if (r >= 36 && r <= 59 && c >= 36 && c <= 59)
        v = (Math.floor((r - 36) / 6) + Math.floor((c - 36) / 6)) % 2 ? 0.9 : 0.1
      return noise > 0 ? v + noise * normal(g) : v
    }),
  )
}

/** Index into [0, n) under a border rule; -1 means "outside, treat as zero". */
function borderIndex(i: number, n: number, border: Border): number {
  if (i >= 0 && i < n) return i
  if (border === 'zero') return -1
  if (border === 'replicate') return i < 0 ? 0 : n - 1
  // Half-sample symmetric reflection: … c b a | a b c … (scipy's 'reflect').
  const period = 2 * n
  const m = ((i % period) + period) % period
  return m < n ? m : period - 1 - m
}

/** Cross-correlation with a kernel of odd size centred on each pixel (a flipped kernel gives convolution). */
export function correlate(img: Image, kernel: Kernel, border: Border = 'reflect'): Image {
  const h = img.length
  const w = img[0].length
  const kh = kernel.length
  const kw = kernel[0].length
  const oy = (kh - 1) / 2
  const ox = (kw - 1) / 2
  const out: Image = Array.from({ length: h }, () => new Array<number>(w).fill(0))
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      let acc = 0
      for (let u = 0; u < kh; u++) {
        const rr = borderIndex(r + u - oy, h, border)
        if (rr < 0) continue
        const row = img[rr]
        const krow = kernel[u]
        for (let v = 0; v < kw; v++) {
          const cc = borderIndex(c + v - ox, w, border)
          if (cc >= 0) acc += krow[v] * row[cc]
        }
      }
      out[r][c] = acc
    }
  }
  return out
}

/** Separable filtering: a row pass with `kx` and a column pass with `ky`. */
export function separable(img: Image, kx: number[], ky: number[], border: Border = 'reflect'): Image {
  return correlate(
    correlate(img, [kx], border),
    ky.map((v) => [v]),
    border,
  )
}

/** Sampled, normalised 1D Gaussian of standard deviation `sigma`, truncated at 3σ. */
export function gaussian1d(sigma: number): number[] {
  const radius = Math.max(1, Math.ceil(3 * sigma))
  const k = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)))
  const s = k.reduce((a, b) => a + b, 0)
  return k.map((v) => v / s)
}

export const gaussianBlur = (img: Image, sigma: number, border: Border = 'reflect') => {
  const g = gaussian1d(sigma)
  return separable(img, g, g, border)
}

/** Sampled Laplacian of Gaussian, shifted to sum to zero so that flat regions give exactly zero. */
export function logKernel(sigma: number): Kernel {
  const radius = Math.max(1, Math.ceil(3 * sigma))
  const s2 = sigma * sigma
  const k = Array.from({ length: 2 * radius + 1 }, (_, i) =>
    Array.from({ length: 2 * radius + 1 }, (_, j) => {
      const r2 = (i - radius) ** 2 + (j - radius) ** 2
      return ((r2 - 2 * s2) / (s2 * s2)) * Math.exp(-r2 / (2 * s2))
    }),
  )
  const n = k.length * k.length
  const m = k.flat().reduce((a, b) => a + b, 0) / n
  // Scale-normalised by σ² so that responses at different σ are comparable.
  return k.map((row) => row.map((v) => (v - m) * s2))
}

export const SOBEL_X: Kernel = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1],
]
export const SOBEL_Y: Kernel = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1],
]

export const mapImage = (img: Image, f: (v: number, r: number, c: number) => number): Image =>
  img.map((row, r) => row.map((v, c) => f(v, r, c)))

export function extent(img: Image): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const row of img)
    for (const v of row) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  return [lo, hi]
}

export const AXIS = Array.from({ length: SIZE }, (_, i) => i)

/** Gaussian smoothing at `sigma` (none if 0) followed by Sobel derivatives divided by 8, so a ramp of slope a gives a. */
export function gradients(img: Image, sigma: number, border: Border = 'reflect'): { gx: Image; gy: Image } {
  const s = sigma > 0 ? gaussianBlur(img, sigma, border) : img
  const gx = mapImage(correlate(s, SOBEL_X, border), (v) => v / 8)
  const gy = mapImage(correlate(s, SOBEL_Y, border), (v) => v / 8)
  return { gx, gy }
}

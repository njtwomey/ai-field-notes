/**
 * Test images (greyscale, row-major [height, width] tensors with values in [0, 1], row 0 at the top) and small binary
 * pattern sets: a checkerboard, gradients, a shapes image, a 5 × 7 digit font with noisy copies, and bars and stripes.
 */

import { normal, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { checkCount, labels, matrix, type Dataset } from './types'

/** A checkerboard of `tile`-pixel squares, `low` and `high` valued, starting with `high` at the top left. */
export function checkerboardImage(size = 64, tile = 8, { low = 0, high = 1 } = {}): Tensor {
  const out = new Float64Array(size * size)
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) out[r * size + c] = (Math.floor(r / tile) + Math.floor(c / tile)) % 2 ? low : high
  return matrix(out, size, size)
}

/**
 * A linear ramp from 0 to 1 across the image in the direction `angle` (radians, 0 = left to right, π/2 = bottom to
 * top); `kind: 'radial'` ramps from 0 at the centre to 1 at the corners.
 */
export function gradientImage(
  size = 64,
  { angle = 0, kind = 'linear' }: { angle?: number; kind?: 'linear' | 'radial' } = {},
): Tensor {
  const out = new Float64Array(size * size)
  const c = (size - 1) / 2
  const [dx, dy] = [Math.cos(angle), Math.sin(angle)]
  // The projection of the corners onto the direction bounds the ramp, so values span exactly [0, 1].
  const reach = c * (Math.abs(dx) + Math.abs(dy)) || 1
  for (let r = 0; r < size; r++)
    for (let col = 0; col < size; col++) {
      const x = col - c
      const y = c - r
      out[r * size + col] =
        kind === 'radial' ? Math.hypot(x, y) / (Math.SQRT2 * c || 1) : 0.5 + (x * dx + y * dy) / (2 * reach)
    }
  return matrix(out, size, size)
}

/**
 * A shapes test image (the site's image-processing notes): on a dark background (0.15), a bright rectangle, a grey
 * disk, a triangle and a checkerboard patch, so straight edges, curved edges, corners and texture all appear. With a
 * stream and `noise`, Gaussian noise of that standard deviation is added.
 */
export function shapesImage(options: { size?: number; noise?: number; stream?: Stream } = {}): Tensor {
  const { size = 64, noise = 0, stream } = options
  if (noise > 0 && !stream) throw new Error('shapesImage: noise needs a stream')
  const k = size / 64
  const out = new Float64Array(size * size)
  for (let row = 0; row < size; row++)
    for (let c = 0; c < size; c++) {
      // Drawn in the original's 64-pixel coordinates with row 0 at the bottom, then flipped so row 0 is the top.
      const r = (size - 1 - row) / k
      const x = c / k
      let v = 0.15
      if (r >= 8 && r <= 27 && x >= 6 && x <= 25) v = 0.9
      if ((r - 46) ** 2 + (x - 16) ** 2 <= 11 ** 2) v = 0.6
      if (r >= 6 && r <= 28 && Math.abs(x - 48) <= (28 - r) * 0.55) v = 0.7
      if (r >= 36 && r <= 59 && x >= 36 && x <= 59)
        v = (Math.floor((r - 36) / 6) + Math.floor((x - 36) / 6)) % 2 ? 0.9 : 0.1
      out[row * size + c] = noise > 0 ? v + noise * normal(stream!) : v
    }
  return matrix(out, size, size)
}

// A 5 × 7 bitmap font for the digits, row by row from the top (the classic HD44780 LCD glyphs).
const FONT: readonly string[][] = [
  ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  ['11111', '00010', '00100', '00010', '00001', '10001', '01110'],
  ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
]

/** The ten digit glyphs as a [10, 7, 5] tensor of 0s and 1s (row 0 at the top). */
export function digitGlyphs(): Tensor {
  const out = new Float64Array(10 * 35)
  FONT.forEach((glyph, d) => glyph.forEach((row, r) => [...row].forEach((ch, c) => (out[d * 35 + r * 5 + c] = +ch))))
  return fromData(out, [10, 7, 5])
}

/**
 * Noisy digits: `perClass` copies of each 5 × 7 glyph, each pixel flipped with probability `flip` and then blurred by
 * Gaussian noise of standard deviation `noise`. x is n × 35 (rows of the image concatenated), y the digit.
 */
export function digits(s: Stream, options: { perClass?: number; flip?: number; noise?: number } = {}): Dataset {
  const { perClass = 20, flip = 0.05, noise = 0.1 } = options
  checkCount(perClass, 'digits')
  const glyphs = digitGlyphs().data
  const n = 10 * perClass
  const x = new Float64Array(n * 35)
  const y = new Int32Array(n)
  for (let d = 0; d < 10; d++)
    for (let k = 0; k < perClass; k++) {
      const row = d * perClass + k
      const r = s.child('digit', d, k)
      for (let p = 0; p < 35; p++) {
        let v = glyphs[d * 35 + p]
        if (r.uniform() < flip) v = 1 - v
        x[row * 35 + p] = noise > 0 ? v + noise * normal(r) : v
      }
      y[row] = d
    }
  return {
    x: matrix(x, n, 35),
    y: labels(y),
    meta: {
      name: 'digits',
      description: `${perClass} noisy copies of each 5 × 7 digit glyph (pixel flip probability ${flip}, noise sd ${noise}).`,
      task: 'images',
      featureNames: Array.from({ length: 35 }, (_, p) => `pixel ${Math.floor(p / 5)},${p % 5}`),
      labelNames: Array.from({ length: 10 }, (_, d) => String(d)),
      stream: s.key,
    },
  }
}

/**
 * Every bars-and-stripes pattern on a size × size grid (MacKay, 2003, "Information Theory, Inference, and Learning
 * Algorithms", §43): each subset of columns switched on (bars) or of rows (stripes), with the all-off and all-on
 * patterns counted once. Returns a [2^{size+1} − 2, size·size] tensor of 0s and 1s.
 */
export function barsAndStripes(size = 4): Tensor {
  const patterns: number[][] = []
  const seen = new Set<string>()
  for (const kind of ['bars', 'stripes'])
    for (let mask = 0; mask < 2 ** size; mask++) {
      const p = Array.from({ length: size * size }, (_, k) => {
        const [r, c] = [Math.floor(k / size), k % size]
        return (mask >> (kind === 'bars' ? c : r)) & 1
      })
      const key = p.join('')
      if (seen.has(key)) continue
      seen.add(key)
      patterns.push(p)
    }
  return matrix(Float64Array.from(patterns.flat()), patterns.length, size * size)
}

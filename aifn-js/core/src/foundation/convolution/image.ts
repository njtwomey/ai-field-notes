/** Two-dimensional correlation and convolution with border modes, part of `aifn/foundation/convolution`. */

import { ShapeError } from 'aifn/foundation/errors'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import { readValues } from 'aifn/foundation/fourier'

/** How samples beyond the edge are read, as scipy.ndimage: `reflect` (d c b a | a b c d), `mirror` (d c b | a b c d), `nearest`, `constant` (zero) or `wrap`. */
export type Border = 'reflect' | 'mirror' | 'nearest' | 'constant' | 'wrap'

/** An image or kernel: a rank-2 tensor or rows. */
export type ImageInput = Tensor | readonly (readonly number[])[]

/** Index into [0, n) under a border mode, or −1 for a constant (zero) sample. */
function borderIndex(i: number, n: number, border: Border): number {
  if (i >= 0 && i < n) return i
  switch (border) {
    case 'constant':
      return -1
    case 'nearest':
      return i < 0 ? 0 : n - 1
    case 'wrap':
      return ((i % n) + n) % n
    case 'reflect': {
      const period = 2 * n
      const m = ((i % period) + period) % period
      return m < n ? m : period - 1 - m
    }
    case 'mirror': {
      if (n === 1) return 0
      const period = 2 * n - 2
      const m = ((i % period) + period) % period
      return m < n ? m : period - m
    }
  }
}

/**
 * Cross-correlation out[r, c] = Σ_{i,j} k[i, j] img[r + i − ⌊kh/2⌋, c + j − ⌊kw/2⌋], as `scipy.ndimage.correlate`
 * (the kernel's centre at ⌊size/2⌋).
 */
export function correlate2d(
  img: ImageInput,
  kernel: ImageInput,
  { border = 'reflect' }: { border?: Border } = {},
): Tensor {
  const I = readImage(img, 'correlate2d')
  const K = readImage(kernel, 'correlate2d kernel')
  const out = new Float64Array(I.h * I.w)
  const [cy, cx] = [Math.floor(K.h / 2), Math.floor(K.w / 2)]
  for (let r = 0; r < I.h; r++)
    for (let c = 0; c < I.w; c++) {
      let s = 0
      for (let i = 0; i < K.h; i++) {
        const rr = borderIndex(r + i - cy, I.h, border)
        if (rr < 0) continue
        for (let j = 0; j < K.w; j++) {
          const cc = borderIndex(c + j - cx, I.w, border)
          if (cc < 0) continue
          s += K.v[i * K.w + j] * I.v[rr * I.w + cc]
        }
      }
      out[r * I.w + c] = s
    }
  return fromData(out, [I.h, I.w])
}

/** Convolution: correlation with the kernel flipped in both axes, as `scipy.ndimage.convolve` (odd kernel sizes). */
export function convolve2d(img: ImageInput, kernel: ImageInput, options: { border?: Border } = {}): Tensor {
  const K = readImage(kernel, 'convolve2d kernel')
  const flipped = K.v.slice().reverse()
  return correlate2d(img, fromData(flipped, [K.h, K.w]), options)
}

/** Separable filtering: correlate each row with kx, then each column with ky. */
export function separableFilter(
  img: ImageInput,
  kx: Tensor | ArrayLike<number>,
  ky: Tensor | ArrayLike<number>,
  options: { border?: Border } = {},
): Tensor {
  const x = readValues(kx)
  const y = readValues(ky)
  const rows = correlate2d(img, fromData(x, [1, x.length]), options)
  return correlate2d(rows, fromData(y, [y.length, 1]), options)
}

/**
 * An image (a 2-D tensor or rows of numbers) as its row-major values, height and width; `what` names the caller in
 * errors. For filters written over raw arrays.
 */
export function readImage(img: ImageInput, what: string): { v: Float64Array; h: number; w: number } {
  if ('shape' in img) {
    if (img.shape.length !== 2) throw new ShapeError(what, `${what}: expected a 2-D array`)
    return { v: readValues(img), h: img.shape[0], w: img.shape[1] }
  }
  const h = img.length
  const w = img[0]?.length ?? 0
  const v = new Float64Array(h * w)
  img.forEach((row, r) => v.set(row, r * w))
  return { v, h, w }
}

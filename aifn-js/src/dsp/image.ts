/**
 * 2-D image operators on greyscale images ([height, width] tensors, row 0 at the top), with scipy.ndimage's boundary
 * modes: correlation and convolution with a kernel, separable filtering, Gaussian blur, Sobel gradients and the
 * Laplacian of Gaussian (Marr and Hildreth, 1980, Proc. R. Soc. Lond. B 207).
 */

import { fromData, type Tensor } from 'aifn/tensor'
import { readValues } from './complex'

/** How samples beyond the edge are read, as scipy.ndimage: `reflect` (d c b a | a b c d), `mirror` (d c b | a b c d), `nearest`, `constant` (zero) or `wrap`. */
export type Border = 'reflect' | 'mirror' | 'nearest' | 'constant' | 'wrap'

/** An image or kernel: a rank-2 tensor or rows. */
export type ImageInput = Tensor | readonly (readonly number[])[]

function readImage(img: ImageInput, what: string): { v: Float64Array; h: number; w: number } {
  if ('shape' in img) {
    if (img.shape.length !== 2) throw new Error(`${what}: expected a 2-D array`)
    return { v: readValues(img), h: img.shape[0], w: img.shape[1] }
  }
  const h = img.length
  const w = img[0]?.length ?? 0
  const v = new Float64Array(h * w)
  img.forEach((row, r) => v.set(row, r * w))
  return { v, h, w }
}

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
 * The sampled, normalised Gaussian kernel exp(−x²/2σ²) on x = −R, …, R with R = ⌊truncate·σ + 1/2⌋, as
 * scipy.ndimage's (truncate default 4).
 */
export function gaussianKernel(sigma: number, { truncate = 4 }: { truncate?: number } = {}): Tensor {
  const radius = Math.floor(truncate * sigma + 0.5)
  const k = Float64Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2))
  const total = k.reduce((a, b) => a + b, 0)
  return fromData(k.map((v) => v / total))
}

/** Gaussian blur with standard deviation σ pixels (separable), as `scipy.ndimage.gaussian_filter`. */
export function gaussianBlur(
  img: ImageInput,
  sigma: number,
  options: { border?: Border; truncate?: number } = {},
): Tensor {
  if (sigma <= 0) {
    const I = readImage(img, 'gaussianBlur')
    return fromData(I.v, [I.h, I.w])
  }
  const k = gaussianKernel(sigma, options)
  return separableFilter(img, k, k, options)
}

/** Sobel gradients. */
export interface Gradients {
  /** Derivative along columns (x, to the right). */
  gx: Tensor
  /** Derivative along rows (downwards, row index increasing). */
  gy: Tensor
  magnitude: Tensor
  /** atan2(gy, gx), radians. */
  direction: Tensor
}

/**
 * Sobel gradients (Sobel and Feldman, 1968): differences [−1, 0, 1] along one axis smoothed by [1, 2, 1] along the
 * other, as `scipy.ndimage.sobel` (unnormalised; divide by 8 for a unit-spacing derivative). With `sigma`, the image is
 * Gaussian-blurred first.
 */
export function sobel(img: ImageInput, options: { border?: Border; sigma?: number } = {}): Gradients {
  const base = options.sigma ? gaussianBlur(img, options.sigma, options) : img
  const gx = separableFilter(base, [-1, 0, 1], [1, 2, 1], options)
  const gy = separableFilter(base, [1, 2, 1], [-1, 0, 1], options)
  const x = gx.data
  const y = gy.data
  return {
    gx,
    gy,
    magnitude: fromData(
      Float64Array.from(x, (v, i) => Math.hypot(v, y[i])),
      gx.shape,
    ),
    direction: fromData(
      Float64Array.from(x, (v, i) => Math.atan2(y[i], v)),
      gx.shape,
    ),
  }
}

/**
 * The Laplacian-of-Gaussian kernel ∇²G_σ, sampled on a (2R + 1)² grid with R = ⌈3σ⌉ and shifted to sum to zero, so
 * that flat regions give zero response.
 */
export function laplacianOfGaussian(sigma: number): Tensor {
  const radius = Math.ceil(3 * sigma)
  const size = 2 * radius + 1
  const k = new Float64Array(size * size)
  let total = 0
  for (let i = 0; i < size; i++)
    for (let j = 0; j < size; j++) {
      const r2 = (i - radius) ** 2 + (j - radius) ** 2
      const v = ((r2 - 2 * sigma * sigma) / sigma ** 4) * Math.exp(-r2 / (2 * sigma * sigma))
      k[i * size + j] = v
      total += v
    }
  const mean = total / (size * size)
  return fromData(
    k.map((v) => v - mean),
    [size, size],
  )
}

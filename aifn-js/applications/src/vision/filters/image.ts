/**
 * 2-D image operators on greyscale images ([height, width] tensors, row 0 at the top), with scipy.ndimage's boundary
 * modes: correlation and convolution with a kernel, separable filtering, Gaussian blur, Sobel gradients and the
 * Laplacian of Gaussian (Marr and Hildreth, 1980, Proc. R. Soc. Lond. B 207).
 */

import { fromData, type Tensor } from 'aifn/foundation/tensor'
import { readImage, separableFilter, type Border, type ImageInput } from 'aifn/foundation/convolution'

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

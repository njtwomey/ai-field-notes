/** The discrete cosine transform (DCT-II, orthonormal), part of `aifn/foundation/fourier`. */

import { fromData, isTensor, type Tensor } from 'aifn/foundation/tensor'
import { readValues } from './complex'

/**
 * The orthonormal DCT-II along the last axis, as `scipy.fft.dct(x, norm='ortho')`:
 * X[k] = √((2 − δ_k0)/N) Σ_n x[n] cos(πk(2n + 1)/(2N)). Direct O(N²) per row, suited to short feature vectors.
 */
export function dct(x: Tensor | ArrayLike<number>): Tensor {
  const v = readValues(x)
  const shape = isTensor(x) ? x.shape : [v.length]
  const n = shape[shape.length - 1]
  const out = new Float64Array(v.length)
  for (let r = 0; r < v.length / n; r++)
    for (let k = 0; k < n; k++) {
      let s = 0
      for (let i = 0; i < n; i++) s += v[r * n + i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n))
      out[r * n + k] = s * Math.sqrt((k === 0 ? 1 : 2) / n)
    }
  return fromData(out, shape)
}

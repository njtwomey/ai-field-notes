import { describe, expect, it } from 'vitest'
import { gaussianBlur, sobel } from 'aifn-applied/vision/filters'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
// The scipy.ndimage references live in core's `signal` fixture (generated with the convolution checks).
import { fixture } from '../../../../core/test/fixtures'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const I = fixture<any>('signal').image

const close = (a: Tensor, e: number[][], tol: number) => {
  const x = toFlat(a)
  const y = e.flat()
  expect(x.length).toBe(y.length)
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol))
}

describe('image filters against scipy.ndimage', () => {
  const img = fromData(Float64Array.from((I.image as number[][]).flat()), [12, 10])
  it('Gaussian blur and Sobel', () => {
    close(gaussianBlur(img, 1.5), I.gaussian, 1e-12)
    const s = sobel(img)
    close(s.gx, I.sobelX, 1e-12)
    close(s.gy, I.sobelY, 1e-12)
  })
})

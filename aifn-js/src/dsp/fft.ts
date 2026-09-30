/**
 * The discrete Fourier transform X[k] = Σₙ x[n] e^{−2πi kn/N} (numpy's sign and scaling: the inverse divides by N).
 * Power-of-two lengths use the iterative radix-2 Cooley–Tukey FFT (Cooley and Tukey, 1965, Math. Comp. 19); other
 * lengths use Bluestein's chirp-z algorithm (Bluestein, 1970, IEEE Trans. Audio Electroacoust. 18(4)), which rewrites a
 * length-N DFT as a circular convolution of power-of-two length ≥ 2N − 1. Both are O(N log N).
 */

import { fromData, isTensor, type Tensor } from 'aifn/tensor'
import { complexOf, readComplex, readSignal, readValues, type ComplexTensor, type Signal } from './complex'

const TAU = 2 * Math.PI

/** True when n is a positive power of two. */
export const isPowerOfTwo = (n: number): boolean => n > 0 && (n & (n - 1)) === 0

/** The smallest power of two ≥ n (1 for n ≤ 1). */
export const nextPowerOfTwo = (n: number): number => 2 ** Math.ceil(Math.log2(Math.max(1, n)))

/** In-place radix-2 FFT; `inverse` flips the sign of the exponent (no 1/N). Length must be a power of two. */
function radix2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]
      re[i] = re[j]
      re[j] = t
      t = im[i]
      im[i] = im[j]
      im[j] = t
    }
  }
  const sign = inverse ? 1 : -1
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1
    // Twiddles computed directly per k (not by repeated multiplication) to keep rounding error O(log n).
    const step = (sign * TAU) / len
    for (let k = 0; k < half; k++) {
      const wr = Math.cos(step * k)
      const wi = Math.sin(step * k)
      for (let start = 0; start < n; start += len) {
        const a = start + k
        const b = a + half
        const tr = re[b] * wr - im[b] * wi
        const ti = re[b] * wi + im[b] * wr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
      }
    }
  }
}

/** In-place Bluestein transform of any length (no 1/N on the inverse). */
function bluestein(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  const m = nextPowerOfTwo(2 * n - 1)
  const sign = inverse ? 1 : -1
  // Chirp w[k] = exp(sign · iπ k²/n); k² is reduced mod 2n so the angle stays small and accurate for large k.
  const cr = new Float64Array(n)
  const ci = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const angle = (sign * Math.PI * ((k * k) % (2 * n))) / n
    cr[k] = Math.cos(angle)
    ci[k] = Math.sin(angle)
  }
  const ar = new Float64Array(m)
  const ai = new Float64Array(m)
  for (let k = 0; k < n; k++) {
    ar[k] = re[k] * cr[k] - im[k] * ci[k]
    ai[k] = re[k] * ci[k] + im[k] * cr[k]
  }
  // b[k] = conj(w[k]) for |k| < n, wrapped circularly.
  const br = new Float64Array(m)
  const bi = new Float64Array(m)
  br[0] = cr[0]
  bi[0] = -ci[0]
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = cr[k]
    bi[k] = bi[m - k] = -ci[k]
  }
  radix2(ar, ai, false)
  radix2(br, bi, false)
  for (let k = 0; k < m; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k]
    const i = ar[k] * bi[k] + ai[k] * br[k]
    ar[k] = r
    ai[k] = i
  }
  radix2(ar, ai, true)
  for (let k = 0; k < n; k++) {
    const r = ar[k] / m
    const i = ai[k] / m
    re[k] = r * cr[k] - i * ci[k]
    im[k] = r * ci[k] + i * cr[k]
  }
}

/** In-place DFT of any length (unscaled inverse). Private to aifn/dsp. */
export function transformInPlace(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (n <= 1) return
  if (isPowerOfTwo(n)) radix2(re, im, inverse)
  else bluestein(re, im, inverse)
}

/** Pad with zeros or truncate to length n. */
function fit(a: Float64Array, n: number): Float64Array {
  if (a.length === n) return a
  const out = new Float64Array(n)
  out.set(a.subarray(0, Math.min(n, a.length)))
  return out
}

/**
 * The DFT of a real or complex rank-1 signal, as `numpy.fft.fft`: X[k] = Σₙ x[n] e^{−2πi kn/N}. `n` zero-pads or
 * truncates the input first (default: its length; any length is fast).
 */
export function fft(x: Signal | ComplexTensor, { n }: { n?: number } = {}): ComplexTensor {
  const input = readComplex(x, 'fft')
  const size = n ?? input.re.length
  const re = fit(input.re, size)
  const im = fit(input.im, size)
  transformInPlace(re, im, false)
  return complexOf(re, im)
}

/** The inverse DFT, as `numpy.fft.ifft`: x[n] = (1/N) Σₖ X[k] e^{2πi kn/N}. */
export function ifft(x: Signal | ComplexTensor, { n }: { n?: number } = {}): ComplexTensor {
  const input = readComplex(x, 'ifft')
  const size = n ?? input.re.length
  const re = fit(input.re, size)
  const im = fit(input.im, size)
  transformInPlace(re, im, true)
  for (let k = 0; k < size; k++) {
    re[k] /= size
    im[k] /= size
  }
  return complexOf(re, im)
}

/** The DFT of a real signal at its non-negative frequencies, k = 0, …, ⌊n/2⌋, as `numpy.fft.rfft`. */
export function rfft(x: Signal, { n }: { n?: number } = {}): ComplexTensor {
  const values = readSignal(x, 'rfft')
  const size = n ?? values.length
  const re = fit(values, size)
  const im = new Float64Array(size)
  transformInPlace(re, im, false)
  const half = Math.floor(size / 2) + 1
  return complexOf(re.slice(0, half), im.slice(0, half))
}

/**
 * The inverse of `rfft`, as `numpy.fft.irfft`: a real signal of length n (default 2(m − 1) for m input bins) whose
 * spectrum has the given non-negative half; the imaginary parts of the DC and Nyquist bins are ignored.
 */
export function irfft(x: ComplexTensor, { n }: { n?: number } = {}): Tensor {
  const { re: hr, im: hi } = readComplex(x, 'irfft')
  const size = n ?? 2 * (hr.length - 1)
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  const half = Math.floor(size / 2)
  for (let k = 0; k <= half && k < hr.length; k++) {
    re[k] = hr[k]
    im[k] = hi[k]
  }
  im[0] = 0
  if (size % 2 === 0 && half < hr.length) im[half] = 0
  // Hermitian symmetry fills the negative frequencies.
  for (let k = 1; k < size - half; k++) {
    re[size - k] = re[k]
    im[size - k] = -im[k]
  }
  transformInPlace(re, im, true)
  for (let k = 0; k < size; k++) re[k] /= size
  return fromData(re)
}

/** The DFT by its definition, O(N²): for checking and for tiny inputs. */
export function dft(x: Signal | ComplexTensor): ComplexTensor {
  const { re: xr, im: xi } = readComplex(x, 'dft')
  const n = xr.length
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let k = 0; k < n; k++)
    for (let t = 0; t < n; t++) {
      const a = (-TAU * ((k * t) % n)) / n
      const c = Math.cos(a)
      const s = Math.sin(a)
      re[k] += xr[t] * c - xi[t] * s
      im[k] += xr[t] * s + xi[t] * c
    }
  return complexOf(re, im)
}

/** Frequencies of the DFT bins, as `numpy.fft.fftfreq`: k/(n d) for k = 0, …, ⌈n/2⌉ − 1, then the negative ones. */
export function fftfreq(n: number, d = 1): Tensor {
  const out = new Float64Array(n)
  for (let k = 0; k < n; k++) out[k] = (k < Math.ceil(n / 2) ? k : k - n) / (n * d)
  return fromData(out)
}

/** Frequencies of the `rfft` bins, k/(n d) for k = 0, …, ⌊n/2⌋, as `numpy.fft.rfftfreq`. */
export function rfftfreq(n: number, d = 1): Tensor {
  return fromData(Float64Array.from({ length: Math.floor(n / 2) + 1 }, (_, k) => k / (n * d)))
}

/** Move the zero-frequency bin to the centre, as `numpy.fft.fftshift` (rank 1). */
export function fftshift(x: Tensor | ArrayLike<number>): Tensor {
  const v = isTensor(x) ? readSignal(x, 'fftshift') : Float64Array.from(x)
  const n = v.length
  const shift = Math.floor(n / 2)
  return fromData(Float64Array.from({ length: n }, (_, i) => v[(i - shift + n) % n]))
}

/** The inverse of `fftshift`, as `numpy.fft.ifftshift`. */
export function ifftshift(x: Tensor | ArrayLike<number>): Tensor {
  const v = isTensor(x) ? readSignal(x, 'ifftshift') : Float64Array.from(x)
  const n = v.length
  const shift = Math.floor(n / 2)
  return fromData(Float64Array.from({ length: n }, (_, i) => v[(i + shift) % n]))
}

/** The 2-D DFT of a real or complex [h, w] array, as `numpy.fft.fft2`: rows, then columns. */
export function fft2(x: Tensor | ComplexTensor, { inverse = false }: { inverse?: boolean } = {}): ComplexTensor {
  const reT = isTensor(x) ? x : x.re
  if (reT.shape.length !== 2) throw new Error('fft2: expected a 2-D array')
  const [h, w] = reT.shape
  const re = readValues(reT)
  const im = isTensor(x) ? new Float64Array(h * w) : readValues(x.im)
  const rowRe = new Float64Array(w)
  const rowIm = new Float64Array(w)
  for (let r = 0; r < h; r++) {
    rowRe.set(re.subarray(r * w, (r + 1) * w))
    rowIm.set(im.subarray(r * w, (r + 1) * w))
    transformInPlace(rowRe, rowIm, inverse)
    re.set(rowRe, r * w)
    im.set(rowIm, r * w)
  }
  const colRe = new Float64Array(h)
  const colIm = new Float64Array(h)
  for (let c = 0; c < w; c++) {
    for (let r = 0; r < h; r++) {
      colRe[r] = re[r * w + c]
      colIm[r] = im[r * w + c]
    }
    transformInPlace(colRe, colIm, inverse)
    for (let r = 0; r < h; r++) {
      re[r * w + c] = colRe[r]
      im[r * w + c] = colIm[r]
    }
  }
  if (inverse)
    for (let k = 0; k < h * w; k++) {
      re[k] /= h * w
      im[k] /= h * w
    }
  return complexOf(re, im, [h, w])
}

/** The inverse 2-D DFT, as `numpy.fft.ifft2`. */
export function ifft2(x: ComplexTensor): ComplexTensor {
  return fft2(x, { inverse: true })
}

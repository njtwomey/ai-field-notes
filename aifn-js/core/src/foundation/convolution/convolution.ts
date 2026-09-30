/**
 * Convolution and cross-correlation of 1-D signals, as `scipy.signal.convolve` and `correlate`, computed directly or
 * by FFT (the convolution theorem: pointwise products of zero-padded transforms, O((n + m) log(n + m))).
 */

import { fromData, type Tensor } from 'aifn/foundation/tensor'
import { readSignal, type Signal } from 'aifn/foundation/fourier'
import { nextPowerOfTwo, transformInPlace } from 'aifn/foundation/fourier'

/** Which part of the full convolution to return: all of it, the centre with the first input's length, or the part without zero padding. */
export type ConvolutionMode = 'full' | 'same' | 'valid'

/** Options for `convolve` and `correlate`. */
export interface ConvolveOptions {
  mode?: ConvolutionMode
  /** `direct` O(nm), `fft`, or `auto` (FFT when n·m is large). Default `auto`. */
  method?: 'direct' | 'fft' | 'auto'
}

function direct(x: Float64Array, h: Float64Array): Float64Array {
  const out = new Float64Array(x.length + h.length - 1)
  for (let i = 0; i < x.length; i++) {
    const xi = x[i]
    if (xi === 0) continue
    for (let j = 0; j < h.length; j++) out[i + j] += xi * h[j]
  }
  return out
}

function viaFft(x: Float64Array, h: Float64Array): Float64Array {
  const n = x.length + h.length - 1
  const size = nextPowerOfTwo(n)
  const ar = new Float64Array(size)
  const ai = new Float64Array(size)
  const br = new Float64Array(size)
  const bi = new Float64Array(size)
  ar.set(x)
  br.set(h)
  transformInPlace(ar, ai)
  transformInPlace(br, bi)
  for (let k = 0; k < size; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k]
    ai[k] = ar[k] * bi[k] + ai[k] * br[k]
    ar[k] = r
  }
  transformInPlace(ar, ai, true)
  return Float64Array.from({ length: n }, (_, k) => ar[k] / size)
}

/** The centred part of a full result, as scipy's `_centered`. */
function centred(full: Float64Array, length: number): Float64Array {
  const start = Math.floor((full.length - length) / 2)
  return full.slice(start, start + length)
}

function convolveRaw(
  x: Float64Array,
  h: Float64Array,
  { mode = 'full', method = 'auto' }: ConvolveOptions,
): Float64Array {
  if (x.length === 0 || h.length === 0) return new Float64Array(0)
  const useFft =
    method === 'fft' || (method === 'auto' && x.length * h.length > 4096 && Math.min(x.length, h.length) > 32)
  const full = useFft ? viaFft(x, h) : direct(x, h)
  if (mode === 'full') return full
  if (mode === 'same') return centred(full, x.length)
  return centred(full, Math.abs(x.length - h.length) + 1)
}

/** The convolution (x ∗ h)[n] = Σ_k x[k] h[n − k], as `scipy.signal.convolve` (full length |x| + |h| − 1 by default). */
export function convolve(x: Signal, h: Signal, options: ConvolveOptions = {}): Tensor {
  return fromData(convolveRaw(readSignal(x, 'convolve'), readSignal(h, 'convolve'), options))
}

/** Convolution by FFT, as `scipy.signal.fftconvolve`. */
export function fftConvolve(x: Signal, h: Signal, { mode = 'full' }: { mode?: ConvolutionMode } = {}): Tensor {
  return convolve(x, h, { mode, method: 'fft' })
}

/**
 * The cross-correlation z[k] = Σ_n x[n + k − (|y| − 1)] y[n] of real signals, as `scipy.signal.correlate`: the
 * convolution of x with y reversed. Use `correlationLags` for the lag of each output.
 */
export function correlate(x: Signal, y: Signal, options: ConvolveOptions = {}): Tensor {
  const yv = readSignal(y, 'correlate').reverse()
  return fromData(convolveRaw(readSignal(x, 'correlate'), yv, options))
}

/** The lags of `correlate(x, y, { mode })` for inputs of lengths n1 and n2, as `scipy.signal.correlation_lags`. */
export function correlationLags(n1: number, n2: number, mode: ConvolutionMode = 'full'): Tensor {
  let lags = Array.from({ length: n1 + n2 - 1 }, (_, i) => i - (n2 - 1))
  if (mode === 'same') {
    const mid = Math.floor(lags.length / 2)
    const bound = Math.floor(n1 / 2)
    lags = n1 % 2 === 0 ? lags.slice(mid - bound, mid + bound) : lags.slice(mid - bound, mid + bound + 1)
  } else if (mode === 'valid') {
    const bound = n1 - n2
    lags =
      bound >= 0
        ? Array.from({ length: bound + 1 }, (_, i) => i)
        : Array.from({ length: 1 - bound }, (_, i) => bound + i)
  }
  return fromData(Int32Array.from(lags))
}

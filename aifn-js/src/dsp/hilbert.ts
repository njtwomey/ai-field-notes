/**
 * The analytic signal and Hilbert spectral analysis: `hilbert` (as `scipy.signal.hilbert`), instantaneous amplitude,
 * phase and frequency, and the Hilbert spectrum of a set of intrinsic mode functions (Huang et al., 1998, Proc. R. Soc.
 * Lond. A 454).
 */

import { fromData, type Tensor } from 'aifn/tensor'
import { complexOf, readSignal, type ComplexTensor, type Signal } from './complex'
import { transformInPlace } from './fft'

/**
 * The analytic signal z = x + i H{x}, as `scipy.signal.hilbert`: the FFT with negative frequencies zeroed and positive
 * ones doubled (DC and Nyquist kept once), inverted. Any length; `n` pads or truncates first.
 */
export function hilbert(x: Signal, { n }: { n?: number } = {}): ComplexTensor {
  const v = readSignal(x, 'hilbert')
  const size = n ?? v.length
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  re.set(v.subarray(0, Math.min(size, v.length)))
  transformInPlace(re, im)
  for (let k = 0; k < size; k++) {
    const h = k === 0 || (size % 2 === 0 && k === size / 2) ? 1 : k < size / 2 ? 2 : 0
    re[k] *= h
    im[k] *= h
  }
  transformInPlace(re, im, true)
  for (let k = 0; k < size; k++) {
    re[k] /= size
    im[k] /= size
  }
  return complexOf(re, im)
}

/** Instantaneous amplitude, phase and frequency of a real signal. */
export interface Instantaneous {
  /** |z|, the envelope. Length n. */
  amplitude: Tensor
  /** Unwrapped arg z, radians. Length n. */
  phase: Tensor
  /**
   * Frequency from the phase increment arg(z[t+1] z̄[t]) · fs / 2π (needs no unwrapping), for t = 0, …, n − 2; the last
   * value repeats so the length is n.
   */
  frequency: Tensor
}

/** Instantaneous amplitude, phase and frequency (in fs units; cycles per sample by default) from `hilbert`. */
export function instantaneous(x: Signal, { fs = 1 }: { fs?: number } = {}): Instantaneous {
  const z = hilbert(x)
  const re = z.re.data as Float64Array
  const im = z.im.data as Float64Array
  const n = re.length
  const amp = new Float64Array(n)
  const ph = new Float64Array(n)
  const freq = new Float64Array(n)
  let offset = 0
  for (let i = 0; i < n; i++) {
    amp[i] = Math.hypot(re[i], im[i])
    if (i + 1 < n) {
      const dr = re[i + 1] * re[i] + im[i + 1] * im[i]
      const di = im[i + 1] * re[i] - re[i + 1] * im[i]
      freq[i] = (Math.atan2(di, dr) * fs) / (2 * Math.PI)
    }
    ph[i] = Math.atan2(im[i], re[i]) + offset
    // Unwrap by accumulating the principal increments.
    if (i > 0) {
      const inc = ph[i] - ph[i - 1]
      if (inc > Math.PI) {
        offset -= 2 * Math.PI
        ph[i] -= 2 * Math.PI
      } else if (inc < -Math.PI) {
        offset += 2 * Math.PI
        ph[i] += 2 * Math.PI
      }
    }
  }
  if (n > 1) freq[n - 1] = freq[n - 2]
  return { amplitude: fromData(amp), phase: fromData(ph), frequency: fromData(freq) }
}

/** The envelope |x + i H{x}| of a real signal. */
export function envelope(x: Signal): Tensor {
  return instantaneous(x).amplitude
}

/** A Hilbert spectrum: amplitude per (frequency bin, time bin), and its marginal over time. */
export interface HilbertSpectrum {
  /** Amplitude summed per cell, shape [freqBins, timeBins], divided by samples per time bin. */
  power: Tensor
  /** h(f) = Σ_t H(t, f). */
  marginal: Tensor
  /** Bin centres. */
  t: Tensor
  f: Tensor
}

/**
 * The Hilbert spectrum H(t, f) of intrinsic mode functions: each mode's instantaneous amplitude is placed at its
 * instantaneous frequency on a grid of timeBins × freqBins cells over [0, fMax] (in fs units), and summed.
 */
export function hilbertSpectrum(
  imfs: readonly Signal[],
  options: { timeBins?: number; freqBins?: number; fMax?: number; fs?: number } = {},
): HilbertSpectrum {
  const { fs = 1, timeBins = 64, freqBins = 64, fMax = fs / 2 } = options
  const tracks = imfs.map((m) => instantaneous(m, { fs }))
  const n = tracks[0]?.amplitude.shape[0] ?? 0
  const per = n / timeBins
  const power = new Float64Array(freqBins * timeBins)
  const marginal = new Float64Array(freqBins)
  for (const { amplitude, frequency } of tracks) {
    const a = amplitude.data
    const f = frequency.data
    for (let i = 0; i < n; i++) {
      const b = Math.floor((f[i] / fMax) * freqBins)
      if (b < 0 || b >= freqBins) continue
      power[b * timeBins + Math.min(timeBins - 1, Math.floor(i / per))] += a[i] / per
      marginal[b] += a[i] / per
    }
  }
  return {
    power: fromData(power, [freqBins, timeBins]),
    marginal: fromData(marginal),
    t: fromData(Float64Array.from({ length: timeBins }, (_, j) => ((j + 0.5) * per) / fs)),
    f: fromData(Float64Array.from({ length: freqBins }, (_, b) => ((b + 0.5) * fMax) / freqBins)),
  }
}

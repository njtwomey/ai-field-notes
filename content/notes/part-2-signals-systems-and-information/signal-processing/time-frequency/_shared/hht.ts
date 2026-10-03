/** Hilbert spectral analysis of IMFs, shared by the Hilbert–Huang and mode-decomposition notes. */
import { analytic } from './tf'

/** Instantaneous amplitude and frequency (cycles per sample) of a real signal whose length is a power of two. */
export function amplitudeFrequency(x: ArrayLike<number>) {
  const z = analytic(x)
  const n = x.length
  const amp = new Float64Array(n)
  const freq = new Float64Array(n)
  for (let i = 0; i < n; i++) amp[i] = Math.hypot(z.re[i], z.im[i])
  // Phase increment arg(z[i+1] z*[i]), which needs no unwrapping; the last sample repeats its neighbour.
  for (let i = 0; i + 1 < n; i++) {
    const re = z.re[i + 1] * z.re[i] + z.im[i + 1] * z.im[i]
    const im = z.im[i + 1] * z.re[i] - z.re[i + 1] * z.im[i]
    freq[i] = Math.atan2(im, re) / (2 * Math.PI)
  }
  freq[n - 1] = freq[n - 2]
  return { amp, freq }
}

/**
 * Hilbert spectrum H(t, f): each IMF's instantaneous amplitude placed at its instantaneous frequency, summed into a
 * grid of `timeBins` × `freqBins` cells over [0, fMax] cycles per sample. Returns z[freq][time] and the marginal
 * spectrum h(f) = Σ_t H(t, f), both divided by the number of samples per time bin.
 */
export function hilbertSpectrum(imfs: ArrayLike<number>[], timeBins: number, freqBins: number, fMax: number) {
  const n = imfs[0]?.length ?? 0
  const per = n / timeBins
  const z = Array.from({ length: freqBins }, () => new Array<number>(timeBins).fill(0))
  const marginal = new Array<number>(freqBins).fill(0)
  const tracks = imfs.map((c) => amplitudeFrequency(c))
  for (const { amp, freq } of tracks) {
    for (let i = 0; i < n; i++) {
      const b = Math.floor((freq[i] / fMax) * freqBins)
      if (b < 0 || b >= freqBins) continue
      z[b][Math.min(timeBins - 1, Math.floor(i / per))] += amp[i] / per
      marginal[b] += amp[i] / per
    }
  }
  const times = Array.from({ length: timeBins }, (_, j) => (j + 0.5) * per)
  const freqs = Array.from({ length: freqBins }, (_, b) => ((b + 0.5) * fMax) / freqBins)
  return { z, marginal, times, freqs, tracks }
}

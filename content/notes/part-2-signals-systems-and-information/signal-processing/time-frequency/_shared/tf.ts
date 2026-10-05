/** Test signals and time-frequency helpers shared by the time-frequency notes. */
import { fft, ifft, isPowerOfTwo, rfft } from 'aifn-compute/foundation/fourier'
import { complex, complexAbs, imagPart, realPart, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { getWindow } from 'aifn-compute/signal/windows'

const TAU = 2 * Math.PI

/**
 * The complex FFT of (re, im), written back into the two arrays. The inverse is unscaled (no 1/N), so callers divide
 * by N where they need the true inverse.
 */
export function fftParts(re: Float64Array, im: Float64Array, inverse = false): void {
  const z = complex(tensor(re), tensor(im))
  const out = inverse ? ifft(z, { norm: 'forward' }) : fft(z)
  re.set(toFlat(realPart(out)))
  im.set(toFlat(imagPart(out)))
}

/** One-sided magnitude spectrum |X[k]|, k = 0 … nfft/2, of x zero-padded to nfft samples. */
export function magnitudeSpectrum(x: ArrayLike<number>, nfft = x.length): number[] {
  const padded = new Float64Array(nfft)
  padded.set(Array.from(x).slice(0, nfft))
  return toFlat(complexAbs(rfft(padded)))
}

/** Decibels, 20 log₁₀ of a magnitude, floored so that zeros stay finite. */
export const db = (magnitude: number, floor = -200) => Math.max(floor, 20 * Math.log10(Math.max(magnitude, 1e-300)))

/**
 * Magnitude spectrogram: Hann-windowed frames of `size` samples every `hop` samples, each zero-padded to `nfft`.
 * Returns each frame's one-sided magnitude spectrum and the frame's centre in samples.
 */
export function spectrogram(x: ArrayLike<number>, size: number, hop: number, nfft = size) {
  const w = toFlat(getWindow('hann', size, { periodic: true }))
  const frames: number[][] = []
  const centres: number[] = []
  for (let start = 0; start + size <= x.length; start += hop) {
    frames.push(
      magnitudeSpectrum(
        Float64Array.from(w, (wi, i) => x[start + i] * wi),
        nfft,
      ),
    )
    centres.push(start + size / 2)
  }
  return { frames, centres }
}

/** Linear chirp cos(2π(f0 t + (f1 − f0) t² / (2T))) over n samples at rate fs, sweeping f0 → f1 Hz. */
export function chirp(n: number, fs: number, f0: number, f1: number, amplitude = 1): Float64Array {
  const duration = n / fs
  const k = (f1 - f0) / duration
  const x = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const t = i / fs
    x[i] = amplitude * Math.cos(TAU * (f0 * t + 0.5 * k * t * t))
  }
  return x
}

/**
 * Analytic signal z = x + i H{x} by zeroing the negative-frequency half of the FFT (length a power of two). Returns
 * real and imaginary parts.
 */
export function analytic(x: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = x.length
  if (!isPowerOfTwo(n)) throw new Error('analytic: length must be a power of two')
  const re = Float64Array.from(x)
  const im = new Float64Array(n)
  fftParts(re, im)
  for (let k = 1; k < n / 2; k++) {
    re[k] *= 2
    im[k] *= 2
  }
  for (let k = n / 2 + 1; k < n; k++) {
    re[k] = 0
    im[k] = 0
  }
  fftParts(re, im, true)
  for (let i = 0; i < n; i++) {
    re[i] /= n
    im[i] /= n
  }
  return { re, im }
}

/**
 * Discrete Wigner–Ville distribution of an analytic signal z (length N, a power of two):
 * W[n, k] = Σ_m z[n+m] z*[n−m] e^{−i2π k (2m) / (2N)}, for lags |m| < min(n+1, N−n, N/2).
 * Row n is time; column k covers frequencies k f_s / (2N), k = 0..N−1, i.e. 0 to just below f_s/2.
 */
export function wignerVille(z: { re: ArrayLike<number>; im: ArrayLike<number> }): Float64Array[] {
  const n = z.re.length
  const rows: Float64Array[] = []
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let t = 0; t < n; t++) {
    re.fill(0)
    im.fill(0)
    const maxLag = Math.min(t, n - 1 - t, n / 2 - 1)
    for (let m = -maxLag; m <= maxLag; m++) {
      // z[t+m] · conj(z[t−m])
      const ar = z.re[t + m]
      const ai = z.im[t + m]
      const br = z.re[t - m]
      const bi = -z.im[t - m]
      const idx = (m + n) % n
      re[idx] = ar * br - ai * bi
      im[idx] = ar * bi + ai * br
    }
    fftParts(re, im)
    // The lag sequence is conjugate-symmetric, so the transform is real.
    rows.push(Float64Array.from(re))
  }
  return rows
}

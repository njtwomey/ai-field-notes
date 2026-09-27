/** Test signals and time-frequency helpers shared by the time-frequency notes. */
import { fftInPlace, isPowerOfTwo } from '@/lib/dsp'

const TAU = 2 * Math.PI

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
  fftInPlace(re, im)
  for (let k = 1; k < n / 2; k++) {
    re[k] *= 2
    im[k] *= 2
  }
  for (let k = n / 2 + 1; k < n; k++) {
    re[k] = 0
    im[k] = 0
  }
  fftInPlace(re, im, true)
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
    fftInPlace(re, im)
    // The lag sequence is conjugate-symmetric, so the transform is real.
    rows.push(Float64Array.from(re))
  }
  return rows
}

/**
 * Small, dependency-free signal-processing helpers for in-browser figures. Sizes here are figure-sized (up to a few
 * thousand samples); anything heavier belongs in a Python `@figure` builder with SciPy.
 */

export type Complex = { re: number; im: number }

const TAU = 2 * Math.PI

/** True when n is a power of two. */
export const isPowerOfTwo = (n: number) => n > 0 && (n & (n - 1)) === 0

/** Smallest power of two ≥ n. */
export const nextPowerOfTwo = (n: number) => 2 ** Math.ceil(Math.log2(Math.max(1, n)))

/**
 * In-place iterative radix-2 FFT of (re, im), length a power of two. `inverse` computes the unscaled inverse; divide
 * by n afterwards (or use `ifft`). X[k] = Σ_n x[n] e^{−2πi kn/N}.
 */
export function fftInPlace(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (!isPowerOfTwo(n)) throw new Error(`fft length ${n} is not a power of two`)
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  const sign = inverse ? 1 : -1
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (sign * TAU) / len
    const wr = Math.cos(angle)
    const wi = Math.sin(angle)
    for (let start = 0; start < n; start += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = start + k
        const b = a + len / 2
        const tr = re[b] * cr - im[b] * ci
        const ti = re[b] * ci + im[b] * cr
        re[b] = re[a] - tr
        im[b] = im[a] - ti
        re[a] += tr
        im[a] += ti
        const next = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = next
      }
    }
  }
}

/** FFT of a real or complex signal, zero-padded to `size` (default: next power of two ≥ length). */
export function fft(x: ArrayLike<number>, size = nextPowerOfTwo(x.length), imag?: ArrayLike<number>) {
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  for (let i = 0; i < Math.min(x.length, size); i++) {
    re[i] = x[i]
    if (imag) im[i] = imag[i]
  }
  fftInPlace(re, im)
  return { re, im }
}

/** Inverse FFT, scaled by 1/N. */
export function ifft(re: ArrayLike<number>, im: ArrayLike<number>) {
  const r = Float64Array.from(re)
  const i = Float64Array.from(im)
  fftInPlace(r, i, true)
  for (let k = 0; k < r.length; k++) {
    r[k] /= r.length
    i[k] /= r.length
  }
  return { re: r, im: i }
}

/** Direct O(N²) DFT of any length, for small N or checking the FFT. */
export function dft(x: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = x.length
  const re = new Float64Array(n)
  const im = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const a = (-TAU * k * t) / n
      re[k] += x[t] * Math.cos(a)
      im[k] += x[t] * Math.sin(a)
    }
  }
  return { re, im }
}

/** |X[k]| for k = 0..N/2 of a real signal (the one-sided magnitude spectrum). */
export function magnitudeSpectrum(x: ArrayLike<number>, size?: number): Float64Array {
  const { re, im } = fft(x, size)
  const half = re.length / 2 + 1
  const out = new Float64Array(half)
  for (let k = 0; k < half; k++) out[k] = Math.hypot(re[k], im[k])
  return out
}

/** Frequencies in Hz of the one-sided spectrum bins for an FFT of `size` at sample rate `fs`. */
export const binFrequencies = (size: number, fs: number) =>
  Array.from({ length: size / 2 + 1 }, (_, k) => (k * fs) / size)

/** Decibels, 20 log₁₀ of a magnitude, floored so zeros stay finite. */
export const db = (magnitude: number, floor = -200) => Math.max(floor, 20 * Math.log10(Math.max(magnitude, 1e-300)))

// Windows. Symmetric by default (filter design); `periodic` gives the DFT-even form used for spectral analysis.

export type WindowName = 'rectangular' | 'hann' | 'hamming' | 'blackman'

export function window(name: WindowName, n: number, periodic = false): Float64Array {
  const w = new Float64Array(n)
  const m = periodic ? n : n - 1
  for (let i = 0; i < n; i++) {
    const t = m === 0 ? 0 : (TAU * i) / m
    w[i] =
      name === 'rectangular'
        ? 1
        : name === 'hann'
          ? 0.5 - 0.5 * Math.cos(t)
          : name === 'hamming'
            ? 0.54 - 0.46 * Math.cos(t)
            : 0.42 - 0.5 * Math.cos(t) + 0.08 * Math.cos(2 * t)
  }
  return w
}

/** Kaiser window with shape parameter β, using the zeroth-order modified Bessel function. */
export function kaiser(n: number, beta: number): Float64Array {
  const i0 = (x: number) => {
    let sum = 1
    let term = 1
    for (let k = 1; k < 50; k++) {
      term *= (x / (2 * k)) ** 2
      sum += term
      if (term < 1e-12 * sum) break
    }
    return sum
  }
  const w = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = n === 1 ? 0 : (2 * i) / (n - 1) - 1
    w[i] = i0(beta * Math.sqrt(1 - r * r)) / i0(beta)
  }
  return w
}

/** Full linear convolution, length |x| + |h| − 1. */
export function convolve(x: ArrayLike<number>, h: ArrayLike<number>): Float64Array {
  const out = new Float64Array(x.length + h.length - 1)
  for (let i = 0; i < x.length; i++) for (let j = 0; j < h.length; j++) out[i + j] += x[i] * h[j]
  return out
}

/**
 * Difference-equation filter, as SciPy's lfilter: a[0] y[n] = Σ b[k] x[n−k] − Σ_{k≥1} a[k] y[n−k], zero initial state.
 */
export function lfilter(b: ArrayLike<number>, a: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const y = new Float64Array(x.length)
  const a0 = a[0]
  for (let n = 0; n < x.length; n++) {
    let acc = 0
    for (let k = 0; k < b.length && k <= n; k++) acc += b[k] * x[n - k]
    for (let k = 1; k < a.length && k <= n; k++) acc -= a[k] * y[n - k]
    y[n] = acc / a0
  }
  return y
}

/**
 * Frequency response H(e^{iω}) = B(e^{iω}) / A(e^{iω}) at `count` frequencies ω in [0, π] (inclusive), as SciPy's
 * freqz with whole = False and endpoint included. Returns ω, magnitude, and unwrapped-free phase in radians.
 */
export function freqz(b: ArrayLike<number>, a: ArrayLike<number> = [1], count = 512) {
  const omega = Array.from({ length: count }, (_, i) => (Math.PI * i) / (count - 1))
  const evaluate = (c: ArrayLike<number>, w: number): Complex => {
    let re = 0
    let im = 0
    for (let k = 0; k < c.length; k++) {
      re += c[k] * Math.cos(-w * k)
      im += c[k] * Math.sin(-w * k)
    }
    return { re, im }
  }
  const magnitude: number[] = []
  const phase: number[] = []
  for (const w of omega) {
    const num = evaluate(b, w)
    const den = evaluate(a, w)
    const d = den.re * den.re + den.im * den.im
    const re = (num.re * den.re + num.im * den.im) / d
    const im = (num.im * den.re - num.re * den.im) / d
    magnitude.push(Math.hypot(re, im))
    phase.push(Math.atan2(im, re))
  }
  return { omega, magnitude, phase }
}

/** Unwrap a phase sequence so consecutive values never jump by more than π. */
export function unwrap(phase: number[]): number[] {
  const out = [...phase]
  for (let i = 1; i < out.length; i++) {
    let d = out[i] - out[i - 1]
    while (d > Math.PI) {
      out[i] -= TAU
      d -= TAU
    }
    while (d < -Math.PI) {
      out[i] += TAU
      d += TAU
    }
  }
  return out
}

/**
 * Short-time Fourier transform magnitudes: frames of `size` samples every `hop` samples, each windowed and FFT'd
 * (padded to `nfft`). Returns frames × (nfft/2 + 1) magnitudes and the frame centre times in samples.
 */
export function stft(
  x: ArrayLike<number>,
  size: number,
  hop: number,
  win: WindowName = 'hann',
  nfft = nextPowerOfTwo(size),
) {
  const w = window(win, size, true)
  const frames: Float64Array[] = []
  const centres: number[] = []
  for (let start = 0; start + size <= x.length; start += hop) {
    const frame = new Float64Array(size)
    for (let i = 0; i < size; i++) frame[i] = x[start + i] * w[i]
    frames.push(magnitudeSpectrum(frame, nfft))
    centres.push(start + size / 2)
  }
  return { frames, centres }
}

/** Hz ↔ mel (HTK formula, 2595 log₁₀(1 + f/700)). */
export const hzToMel = (f: number) => 2595 * Math.log10(1 + f / 700)
export const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1)

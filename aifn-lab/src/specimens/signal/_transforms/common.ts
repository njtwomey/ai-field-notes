/**
 * Small helpers shared by the filters-and-transforms pages: stems for discrete sequences, one-sided DFT amplitude
 * spectra, SNR, seeded noise and raster rows. Signal processing itself comes from aifn; these only shape its results
 * for the charts.
 */
import { rfft } from 'aifn/foundation/fourier'
import { normals, stream } from 'aifn/foundation/random'
import { toComplexFlat, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { getWindow, type WindowSpec } from 'aifn/signal/windows'

/** Vertical segments from (xᵢ, 0) to (xᵢ, yᵢ): a stem plot's stalks (draw `Points` for the heads). */
export function stems(x: ArrayLike<number>, y: ArrayLike<number>, base = 0) {
  return Array.from({ length: x.length }, (_, i) => ({
    from: [x[i], base] as const,
    to: [x[i], y[i]] as const,
  }))
}

/**
 * The one-sided amplitude spectrum of x under a window, scaled so that a sinusoid A cos(2πfn + φ) centred on a bin reads
 * A at that bin (2|X_k| / Σw; DC and Nyquist once), at nfft = pad·N points; with its phase in radians and frequencies in
 * cycles per sample.
 */
export function amplitudeSpectrum(x: ArrayLike<number>, window: WindowSpec = 'rectangular', pad = 1) {
  const n = x.length
  const w = toFlat(getWindow(window, n, { periodic: true }))
  const gain = w.reduce((a, b) => a + b, 0)
  const nfft = n * pad
  const X = toComplexFlat(
    rfft(
      Float64Array.from({ length: n }, (_, i) => x[i] * w[i]),
      { n: nfft },
    ) as Tensor,
  )
  const f = X.map((_, k) => k / nfft)
  const amp = X.map((z, k) => ((k === 0 || 2 * k === nfft ? 1 : 2) * Math.hypot(z.re, z.im)) / gain)
  const phase = X.map((z) => Math.atan2(z.im, z.re))
  return { f, amp, phase, X }
}

/** 10 log₁₀ of signal power over error power, in dB. */
export function snrDb(clean: ArrayLike<number>, estimate: ArrayLike<number>): number {
  let s = 0
  let e = 0
  for (let i = 0; i < clean.length; i++) {
    s += clean[i] ** 2
    e += (clean[i] - estimate[i]) ** 2
  }
  return 10 * Math.log10(s / Math.max(e, 1e-300))
}

/** n seeded standard-normal draws times sd (the same draws for the same name). */
export function noise(name: string, n: number, sd = 1): number[] {
  return toFlat(normals(stream(name), n, 0, sd))
}

/** The rows of a [r, c] tensor, for a `Raster`. */
export function rows(t: Tensor): number[][] {
  const [r, c] = t.shape
  const v = toFlat(t)
  return Array.from({ length: r }, (_, i) => v.slice(i * c, (i + 1) * c))
}

/** Decibels 20 log₁₀ |v| floored at `floor` dB. */
export const db = (v: number, floor = -120) => Math.max(floor, 20 * Math.log10(Math.max(Math.abs(v), 1e-300)))

/** 0, 1, …, n − 1. */
export const range = (n: number) => Array.from({ length: n }, (_, i) => i)

/** Fixed number formatting for readouts. */
export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? v.toFixed(digits) : '—')

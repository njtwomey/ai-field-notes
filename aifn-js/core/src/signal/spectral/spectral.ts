/**
 * Spectral estimation and short-time transforms with scipy.signal's conventions: `periodogram`, `welch` (Welch, 1967,
 * IEEE Trans. Audio Electroacoust. 15(2)), `spectrogram`, `stft`, and Thomson's multitaper estimate (Thomson, 1982,
 * Proc. IEEE 70(9)) with discrete prolate spheroidal sequences (Slepian, 1978, Bell System Tech. J. 57(5)).
 *
 * Densities are one-sided by default: power at frequencies 0 < f < fs/2 is doubled, so the PSD integrates over
 * [0, fs/2] to the signal's variance (mean power after detrending). Estimates are `Spectrum`s and short-time
 * transforms `TimeFrequency` rasters; a `Signal` input supplies fs, which an `fs` option overrides.
 */

import { fromData, isTensor, type Tensor } from 'aifn/foundation/tensor'
import { rfftfreq, transformInPlace } from 'aifn/foundation/fourier'
import type { Scalar, Size, Spectrum, TimeFrequency } from 'aifn/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn/signal/windows'
import { complexValues, powerUnit, readSamples, spectrum, timeFrequency, type SignalInput } from '../signal'

/** Detrending of each segment before its transform. */
export type Detrend = 'constant' | 'linear' | false

/** Options shared by the segment-based estimators. */
export interface SegmentOptions {
  /** Sampling frequency. Default: the signal's, or 1 (frequencies in cycles per sample) for bare samples. */
  fs?: Scalar
  /** Window spec or explicit values of length `nperseg`; built periodic (DFT-even), as scipy does. */
  window?: WindowInput
  /** Samples per segment. */
  nperseg?: number
  /** Samples shared by consecutive segments. */
  noverlap?: number
  /** FFT length (≥ nperseg; zero-padded). Default nperseg. */
  nfft?: number
  detrend?: Detrend
  /** `density` (power per Hz, V²/Hz) or `spectrum` (power per bin, V²). Default `density`. */
  scaling?: 'density' | 'spectrum'
  /** One-sided spectrum for real input. Default true. */
  onesided?: boolean
}

function detrendInPlace(seg: Float64Array, kind: Detrend): void {
  const n = seg.length
  if (!kind || n === 0) return
  let mean = 0
  for (let i = 0; i < n; i++) mean += seg[i]
  mean /= n
  if (kind === 'constant') {
    for (let i = 0; i < n; i++) seg[i] -= mean
    return
  }
  // Least-squares line through (i, seg[i]).
  const tMean = (n - 1) / 2
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += (i - tMean) * (seg[i] - mean)
    sxx += (i - tMean) ** 2
  }
  const slope = sxx > 0 ? sxy / sxx : 0
  for (let i = 0; i < n; i++) seg[i] -= mean + slope * (i - tMean)
}

interface Segmented {
  freqs: Float64Array
  times: Float64Array
  /** Per segment: real and imaginary parts of the scaled, windowed transform (length nfreq each). */
  re: Float64Array[]
  im: Float64Array[]
  nfreq: number
  scale: number
}

/**
 * scipy.signal's `_spectral_helper` for real input: split into segments of `nperseg` stepping by nperseg − noverlap,
 * detrend, window, and take the FFT. Returns the unscaled transforms and the scale factor for the requested scaling.
 */
function segmentTransforms(x: Float64Array, o: Required<Omit<SegmentOptions, 'window'>> & { window: WindowInput }) {
  const { fs, nperseg, noverlap, nfft, detrend, scaling, onesided } = o
  if (nperseg < 1 || nperseg > x.length) throw new RangeError(`nperseg must be in [1, ${x.length}], got ${nperseg}`)
  if (noverlap < 0 || noverlap >= nperseg) throw new RangeError(`noverlap must be in [0, nperseg), got ${noverlap}`)
  if (nfft < nperseg) throw new RangeError('nfft must be at least nperseg')
  const win = windowValues(o.window, nperseg, true)
  let sumW = 0
  let sumW2 = 0
  for (const w of win) {
    sumW += w
    sumW2 += w * w
  }
  const scale = scaling === 'density' ? 1 / (fs * sumW2) : 1 / (sumW * sumW)
  const step = nperseg - noverlap
  const count = Math.floor((x.length - noverlap) / step)
  const nfreq = onesided ? Math.floor(nfft / 2) + 1 : nfft
  const out: Segmented = {
    freqs: onesided
      ? (rfftfreq(nfft, 1 / fs).data as Float64Array)
      : Float64Array.from({ length: nfft }, (_, k) => (k < Math.ceil(nfft / 2) ? k : k - nfft) * (fs / nfft)),
    times: Float64Array.from({ length: count }, (_, k) => (nperseg / 2 + k * step) / fs),
    re: [],
    im: [],
    nfreq,
    scale,
  }
  const seg = new Float64Array(nperseg)
  for (let k = 0; k < count; k++) {
    seg.set(x.subarray(k * step, k * step + nperseg))
    detrendInPlace(seg, detrend)
    const re = new Float64Array(nfft)
    const im = new Float64Array(nfft)
    for (let i = 0; i < nperseg; i++) re[i] = seg[i] * win[i]
    transformInPlace(re, im)
    out.re.push(re.subarray(0, nfreq))
    out.im.push(im.subarray(0, nfreq))
  }
  return out
}

/** Power per segment, scaled, with one-sided doubling of the bins strictly between DC and Nyquist. */
function segmentPowers(s: Segmented, nfft: number, onesided: boolean): Float64Array[] {
  return s.re.map((re, k) => {
    const im = s.im[k]
    const p = new Float64Array(s.nfreq)
    for (let j = 0; j < s.nfreq; j++) {
      p[j] = (re[j] * re[j] + im[j] * im[j]) * s.scale
      const nyquist = nfft % 2 === 0 && j === nfft / 2
      if (onesided && j > 0 && !nyquist) p[j] *= 2
    }
    return p
  })
}

function resolve(
  x: Float64Array,
  fs: Scalar,
  o: SegmentOptions,
  defaults: { window: WindowInput; nperseg: number; overlap: (n: number) => number; detrend: Detrend },
) {
  const nperseg = Math.min(o.nperseg ?? defaults.nperseg, x.length)
  return {
    fs,
    window: o.window ?? defaults.window,
    nperseg,
    noverlap: o.noverlap ?? defaults.overlap(nperseg),
    nfft: o.nfft ?? nperseg,
    detrend: o.detrend ?? defaults.detrend,
    scaling: o.scaling ?? 'density',
    onesided: o.onesided ?? true,
  }
}

/** The name of a window input, for readouts: its spec name, or `custom` for explicit values. */
function windowName(w: WindowInput): string {
  if (typeof w === 'string') return w
  if (!isTensor(w) && typeof w === 'object' && 'name' in w) return (w as { name: string }).name
  return 'custom'
}

/** A power `Spectrum` of a real signal from one-sided (or two-sided) frequencies and values. */
function powerSpectrum(
  f: ArrayLike<number>,
  values: ArrayLike<number>,
  o: { fs: Scalar; scaling: 'density' | 'spectrum'; onesided: boolean },
  unit: string | undefined,
): Spectrum {
  const u = powerUnit(unit, o.scaling === 'density')
  return spectrum({
    f: fromData(Float64Array.from(f), [f.length]),
    axis: 'hz',
    values: fromData(Float64Array.from(values), [values.length]),
    quantity: o.scaling === 'density' ? 'psd' : 'power',
    sided: o.onesided ? 'one' : 'two',
    fs: o.fs,
    ...(u !== undefined ? { unit: u } : {}),
  })
}

/**
 * Welch's estimate of the power spectral density, as `scipy.signal.welch`: the average (mean or median) of windowed,
 * detrended periodograms of overlapping segments. Defaults: Hann window, 256-sample segments (or the whole signal if
 * shorter), 50% overlap, constant detrending.
 */
export function welch(
  x: SignalInput,
  options: SegmentOptions & { average?: 'mean' | 'median' } = {},
): Spectrum & { segments: Size } {
  const input = readSamples(x, 'welch', options.fs)
  const v = input.values
  const o = resolve(v, input.fs, options, {
    window: 'hann',
    nperseg: 256,
    overlap: (n) => Math.floor(n / 2),
    detrend: 'constant',
  })
  const seg = segmentTransforms(v, o)
  const powers = segmentPowers(seg, o.nfft, o.onesided)
  const psd = new Float64Array(seg.nfreq)
  if ((options.average ?? 'mean') === 'mean') {
    for (const p of powers) for (let j = 0; j < psd.length; j++) psd[j] += p[j] / powers.length
  } else {
    // scipy divides the median by the bias of the median of χ²₂ variables, Σ_{i=1}^{K} (−1)^{i+1}/i for odd K.
    const k = powers.length
    let bias = 0
    for (let i = 1; i <= (k % 2 ? k : k - 1); i++) bias += (i % 2 ? 1 : -1) / i
    for (let j = 0; j < psd.length; j++) {
      const col = powers.map((p) => p[j]).sort((a, b) => a - b)
      const med = k % 2 ? col[(k - 1) / 2] : (col[k / 2 - 1] + col[k / 2]) / 2
      psd[j] = med / bias
    }
  }
  return { ...powerSpectrum(seg.freqs, psd, o, input.unit), segments: powers.length }
}

/**
 * The periodogram, as `scipy.signal.periodogram`: one segment (the whole signal), rectangular window by default,
 * constant detrending. With a window it is the modified periodogram.
 */
export function periodogram(x: SignalInput, options: Omit<SegmentOptions, 'nperseg' | 'noverlap'> = {}): Spectrum {
  const n = readSamples(x, 'periodogram').values.length
  const { segments: _segments, ...r } = welch(x, {
    window: 'boxcar',
    ...options,
    nperseg: n,
    noverlap: 0,
    nfft: Math.max(options.nfft ?? n, n),
  })
  return r
}

/**
 * The spectrogram, as `scipy.signal.spectrogram` (mode 'psd'): the scaled periodogram of each segment, arranged
 * frequency × time, as a `TimeFrequency` raster (`method: 'stft'`, `quantity: 'power'`; times are segment centres
 * after the signal's t0). Defaults: Tukey(0.25) window, 256-sample segments, 1/8 overlap, constant detrending.
 */
export function spectrogram(x: SignalInput, options: SegmentOptions = {}): TimeFrequency {
  const input = readSamples(x, 'spectrogram', options.fs)
  const v = input.values
  const o = resolve(v, input.fs, options, {
    window: { name: 'tukey', alpha: 0.25 },
    nperseg: 256,
    overlap: (n) => Math.floor(n / 8),
    detrend: 'constant',
  })
  const seg = segmentTransforms(v, o)
  const powers = segmentPowers(seg, o.nfft, o.onesided)
  const T = powers.length
  const out = new Float64Array(seg.nfreq * T)
  powers.forEach((p, t) => p.forEach((val, j) => (out[j * T + t] = val)))
  return timeFrequency({
    t: fromData(
      seg.times.map((t) => t + input.t0),
      [T],
    ),
    f: fromData(seg.freqs, [seg.nfreq]),
    values: fromData(out, [seg.nfreq, T]),
    quantity: 'power',
    method: 'stft',
    frequencyScale: 'linear',
    window: { name: windowName(o.window), length: o.nperseg, hop: o.nperseg - o.noverlap },
  })
}

/**
 * The short-time Fourier transform, as `scipy.signal.stft`: the signal is padded with nperseg/2 zeros at both ends
 * (so the first and last segments are centred on the ends) and at the end to a whole number of steps, and each
 * windowed segment's transform is divided by the window's sum. Defaults: Hann window, 256 samples, 50% overlap, no
 * detrending. Returns a `TimeFrequency` raster with complex values [f, t, 2] (the interim complex layout).
 */
export function stft(
  x: SignalInput,
  options: Omit<SegmentOptions, 'scaling'> & { boundary?: boolean; padded?: boolean } = {},
): TimeFrequency {
  const input = readSamples(x, 'stft', options.fs)
  const v0 = input.values
  const nperseg = options.nperseg ?? 256
  const noverlap = options.noverlap ?? Math.floor(nperseg / 2)
  const boundary = options.boundary ?? true
  let v = v0
  if (boundary) {
    const ext = Math.floor(nperseg / 2)
    v = new Float64Array(v0.length + 2 * ext)
    v.set(v0, ext)
  }
  if (options.padded ?? true) {
    const step = nperseg - noverlap
    const nadd = (((-(v.length - nperseg) % step) + step) % step) % nperseg
    if (nadd > 0) {
      const w = new Float64Array(v.length + nadd)
      w.set(v)
      v = w
    }
  }
  const fs = input.fs
  const o = {
    fs,
    window: options.window ?? 'hann',
    nperseg,
    noverlap,
    nfft: options.nfft ?? nperseg,
    detrend: options.detrend ?? false,
    scaling: 'spectrum' as const,
    onesided: options.onesided ?? true,
  }
  const seg = segmentTransforms(v, o)
  const scale = Math.sqrt(seg.scale)
  const T = seg.re.length
  const re = new Float64Array(seg.nfreq * T)
  const im = new Float64Array(seg.nfreq * T)
  for (let t = 0; t < T; t++)
    for (let j = 0; j < seg.nfreq; j++) {
      re[j * T + t] = seg.re[t][j] * scale
      im[j * T + t] = seg.im[t][j] * scale
    }
  const shift = boundary ? nperseg / 2 / fs : 0
  return timeFrequency({
    t: fromData(
      seg.times.map((t) => t - shift + input.t0),
      [T],
    ),
    f: fromData(seg.freqs, [seg.nfreq]),
    values: complexValues(re, im, [seg.nfreq, T]),
    quantity: 'complex',
    method: 'stft',
    frequencyScale: 'linear',
    window: { name: windowName(o.window), length: nperseg, hop: nperseg - noverlap },
  })
}

// ── Multitaper ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Eigenpairs of the largest `k` eigenvalues of a symmetric tridiagonal matrix (diagonal d, off-diagonal e), by
 * Sturm-sequence bisection for the eigenvalues and inverse iteration for the vectors (Golub and Van Loan, 2013,
 * "Matrix Computations", §8.4.1–8.4.2). O(n k) per iteration; suited to the well-separated DPSS spectrum.
 */
function tridiagonalTop(d: Float64Array, e: Float64Array, k: number): { values: number[]; vectors: Float64Array[] } {
  const n = d.length
  // Number of eigenvalues less than x (Sturm count).
  const below = (x: number) => {
    let count = 0
    let q = d[0] - x
    if (q < 0) count++
    for (let i = 1; i < n; i++) {
      q = d[i] - x - (e[i - 1] * e[i - 1]) / (q === 0 ? 1e-300 : q)
      if (q < 0) count++
    }
    return count
  }
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < n; i++) {
    const r = (i > 0 ? Math.abs(e[i - 1]) : 0) + (i < n - 1 ? Math.abs(e[i]) : 0)
    lo = Math.min(lo, d[i] - r)
    hi = Math.max(hi, d[i] + r)
  }
  const values: number[] = []
  const vectors: Float64Array[] = []
  for (let j = 0; j < k; j++) {
    // The (n − 1 − j)-th smallest eigenvalue: the smallest x with below(x) ≥ n − j.
    let a = lo
    let b = hi
    for (let it = 0; it < 200 && b - a > 1e-14 * Math.max(1, Math.abs(a) + Math.abs(b)); it++) {
      const mid = (a + b) / 2
      if (below(mid) >= n - j) b = mid
      else a = mid
    }
    const lambda = (a + b) / 2
    values.push(lambda)
    // Inverse iteration with a slightly perturbed shift; three solves suffice for separated eigenvalues.
    const shift = lambda + 1e-10 * Math.max(1, Math.abs(lambda))
    let v: Float64Array = new Float64Array(n).fill(1 / Math.sqrt(n))
    for (let it = 0; it < 3; it++) {
      v = solveTridiagonal(d, e, shift, v)
      // Orthogonalise against the vectors already found, in case of close eigenvalues.
      for (const u of vectors) {
        let dot = 0
        for (let i = 0; i < n; i++) dot += u[i] * v[i]
        for (let i = 0; i < n; i++) v[i] -= dot * u[i]
      }
      let norm = 0
      for (let i = 0; i < n; i++) norm += v[i] * v[i]
      norm = Math.sqrt(norm)
      for (let i = 0; i < n; i++) v[i] /= norm
    }
    vectors.push(v)
  }
  return { values, vectors }
}

/** Solve (T − σI) y = b for tridiagonal T, by Gaussian elimination with partial pivoting (LAPACK's gttrf/gttrs). */
function solveTridiagonal(d: Float64Array, e: Float64Array, sigma: number, b: Float64Array): Float64Array {
  const n = d.length
  // Rows as (sub, diag, sup, sup2) after pivoting.
  const dl = Float64Array.from(e)
  const dd = d.map((v) => v - sigma)
  const du = Float64Array.from(e)
  const du2 = new Float64Array(n)
  const y = Float64Array.from(b)
  const piv = new Uint8Array(n)
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(dd[i]) >= Math.abs(dl[i])) {
      const f = dd[i] === 0 ? 0 : dl[i] / dd[i]
      dl[i] = f
      dd[i + 1] -= f * du[i]
      y[i + 1] -= f * y[i]
    } else {
      piv[i] = 1
      const f = dd[i] / dl[i]
      dd[i] = dl[i]
      dl[i] = f
      const t = du[i]
      du[i] = dd[i + 1]
      dd[i + 1] = t - f * dd[i + 1]
      if (i < n - 2) {
        du2[i] = du[i + 1]
        du[i + 1] = -f * du[i + 1]
      }
      const yt = y[i]
      y[i] = y[i + 1]
      y[i + 1] = yt - f * y[i + 1]
    }
  }
  const tiny = 1e-300
  const x = new Float64Array(n)
  x[n - 1] = y[n - 1] / (dd[n - 1] || tiny)
  if (n > 1) x[n - 2] = (y[n - 2] - du[n - 2] * x[n - 1]) / (dd[n - 2] || tiny)
  for (let i = n - 3; i >= 0; i--) x[i] = (y[i] - du[i] * x[i + 1] - du2[i] * x[i + 2]) / (dd[i] || tiny)
  return x
}

/** Discrete prolate spheroidal sequences. */
export interface Dpss {
  /** The tapers, k × n, each of unit energy. */
  tapers: Tensor
  /** Concentration ratios λ: the fraction of each taper's energy inside the band |f| < W. */
  concentrations: Tensor
}

/**
 * The first k discrete prolate spheroidal (Slepian) sequences of length n and time–half-bandwidth product NW, as
 * `scipy.signal.windows.dpss(n, NW, k)`: the leading eigenvectors of the tridiagonal matrix that commutes with the
 * concentration operator (Percival and Walden, 1993, "Spectral Analysis for Physical Applications", §8.3). Unit
 * energy; even tapers have a positive sum, odd tapers start with a positive lobe (Percival and Walden, p. 379).
 */
export function dpss(n: Size, nw: Scalar, k: Size): Dpss {
  if (!(k >= 1 && k <= n)) throw new RangeError(`dpss: need 1 ≤ k ≤ n, got k = ${k}`)
  const w = nw / n
  const d = Float64Array.from({ length: n }, (_, i) => ((n - 1 - 2 * i) / 2) ** 2 * Math.cos(2 * Math.PI * w))
  const e = Float64Array.from({ length: n - 1 }, (_, i) => ((i + 1) * (n - i - 1)) / 2)
  const { vectors } = tridiagonalTop(d, e, k)
  const thresh = Math.max(1e-7, 1 / n)
  vectors.forEach((v, order) => {
    let flip: boolean
    if (order % 2 === 0) flip = v.reduce((a, b) => a + b, 0) < 0
    else {
      const first = v.find((x) => x * x > thresh) ?? 0
      flip = first < 0
    }
    if (flip) for (let i = 0; i < n; i++) v[i] = -v[i]
  })
  // λ = Σ_l r[l] sin(2πWl)/(πl) over lags, with r the taper's autocorrelation (computed by FFT).
  const m = 2 * n
  const conc = vectors.map((v) => {
    const re = new Float64Array(m)
    const im = new Float64Array(m)
    re.set(v)
    transformInPlace(re, im)
    for (let j = 0; j < m; j++) {
      re[j] = re[j] * re[j] + im[j] * im[j]
      im[j] = 0
    }
    transformInPlace(re, im, true)
    let lambda = (re[0] / m) * 2 * w
    for (let l = 1; l < n; l++) lambda += (2 * (re[l] / m) * Math.sin(2 * Math.PI * w * l)) / (Math.PI * l)
    return lambda
  })
  const flat = new Float64Array(k * n)
  vectors.forEach((v, j) => flat.set(v, j * n))
  return { tapers: fromData(flat, [k, n]), concentrations: fromData(Float64Array.from(conc)) }
}

/**
 * Thomson's multitaper PSD: the average of the k eigenspectra |Σₙ v_j[n] x[n] e^{−2πi fn}|² / fs, one-sided (doubled
 * off DC and Nyquist), with DPSS tapers of time–half-bandwidth NW. Default k = 2NW − 1, the tapers that are well
 * concentrated.
 */
export function multitaper(
  x: SignalInput,
  options: { nw?: Scalar; k?: Size; fs?: Scalar; nfft?: Size; detrend?: Detrend } = {},
): Spectrum & Dpss {
  const input = readSamples(x, 'multitaper', options.fs)
  const v = input.values
  const n = v.length
  const fs = input.fs
  const { nw = 4, detrend = 'constant' } = options
  const k = options.k ?? Math.max(1, Math.floor(2 * nw) - 1)
  const nfft = Math.max(options.nfft ?? n, n)
  detrendInPlace(v, detrend)
  const tapers = dpss(n, nw, k)
  const nfreq = Math.floor(nfft / 2) + 1
  const psd = new Float64Array(nfreq)
  for (let j = 0; j < k; j++) {
    const re = new Float64Array(nfft)
    const im = new Float64Array(nfft)
    for (let i = 0; i < n; i++) re[i] = v[i] * tapers.tapers.data[j * n + i]
    transformInPlace(re, im)
    for (let b = 0; b < nfreq; b++) psd[b] += (re[b] * re[b] + im[b] * im[b]) / (k * fs)
  }
  for (let b = 1; b < nfreq; b++) if (!(nfft % 2 === 0 && b === nfft / 2)) psd[b] *= 2
  const f = rfftfreq(nfft, 1 / fs).data
  return { ...powerSpectrum(f, psd, { fs, scaling: 'density', onesided: true }, input.unit), ...tapers }
}

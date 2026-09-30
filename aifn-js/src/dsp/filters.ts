/**
 * Digital filters with scipy.signal's conventions: window-method FIR design (`firwin`, Kaiser's formulas), IIR design
 * from analog prototypes by the bilinear transform (`iirfilter`, `butter`, `cheby1`, `cheby2`; Oppenheim and Schafer,
 * 2010, "Discrete-Time Signal Processing", §7.1–7.3), filtering by the transposed direct form II (`lfilter`), zero-phase
 * filtering (`filtfilt`, Gustafsson's initial conditions as in scipy), and frequency and group-delay responses.
 *
 * Frequencies: without `fs`, cutoffs are fractions of the Nyquist frequency in (0, 1), as in scipy; with `fs`, they
 * are in the same units as `fs`.
 */

import { solve } from 'aifn/linalg'
import { fromData, type Tensor } from 'aifn/tensor'
import { complexOf, readSignal, readValues, type ComplexTensor, type Signal } from './complex'
import { getWindow, type WindowSpec } from './windows'

// ── Complex numbers (private) ─────────────────────────────────────────────────────────────────────────────────────

type C = { re: number; im: number }
const c = (re: number, im = 0): C => ({ re, im })
const cadd = (a: C, b: C): C => c(a.re + b.re, a.im + b.im)
const csub = (a: C, b: C): C => c(a.re - b.re, a.im - b.im)
const cmul = (a: C, b: C): C => c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
const cscale = (a: C, k: number): C => c(a.re * k, a.im * k)
const cdiv = (a: C, b: C): C => {
  const d = b.re * b.re + b.im * b.im
  return c((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
}
const csqrt = (a: C): C => {
  const r = Math.hypot(a.re, a.im)
  const re = Math.sqrt((r + a.re) / 2)
  const im = Math.sqrt(Math.max(0, (r - a.re) / 2))
  return c(re, a.im < 0 ? -im : im)
}
const cexp = (a: C): C => cscale(c(Math.cos(a.im), Math.sin(a.im)), Math.exp(a.re))
const prod = (xs: C[]): C => xs.reduce(cmul, c(1))

/** Coefficients (highest power first) of Π (z − r); real parts only, since the roots come in conjugate pairs. */
function poly(roots: C[]): number[] {
  let p: C[] = [c(1)]
  for (const r of roots) {
    const next: C[] = [...p, c(0)]
    for (let i = 0; i < p.length; i++) next[i + 1] = csub(next[i + 1], cmul(r, p[i]))
    p = next
  }
  return p.map((v) => v.re)
}

// ── FIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))

/** Options for `firwin`. */
export interface FirwinOptions {
  /** Window for the ideal impulse response. Default hamming. */
  window?: WindowSpec
  /**
   * The band type, or whether the DC gain is 1: `true`/`lowpass`/`bandstop` pass zero frequency, `false`/`highpass`/
   * `bandpass` do not. Default true.
   */
  passZero?: boolean | 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** Scale so the gain is exactly 1 at the centre of the first passband. Default true. */
  scale?: boolean
  fs?: number
}

/**
 * A linear-phase FIR filter by the window method, as `scipy.signal.firwin`: the ideal band-pass impulse response
 * Σ (f_hi sinc(f_hi m) − f_lo sinc(f_lo m)) over the passbands, times a window. `cutoff` is one edge or a list of band
 * edges. An odd number of taps is required when the filter passes the Nyquist frequency.
 */
export function firwin(numtaps: number, cutoff: number | readonly number[], options: FirwinOptions = {}): Tensor {
  const { window = 'hamming', scale = true } = options
  const nyq = (options.fs ?? 2) / 2
  const edges = (typeof cutoff === 'number' ? [cutoff] : [...cutoff]).map((f) => f / nyq)
  if (edges.some((f) => !(f > 0 && f < 1)))
    throw new RangeError('firwin: cutoffs must lie strictly between 0 and Nyquist')
  const pz = options.passZero ?? true
  const passZero = pz === true || pz === 'lowpass' || pz === 'bandstop'
  const passNyquist = (edges.length % 2 === 1) !== passZero
  if (passNyquist && numtaps % 2 === 0)
    throw new RangeError('firwin: a filter that passes the Nyquist frequency needs an odd number of taps')
  const bands = [...(passZero ? [0] : []), ...edges, ...(passNyquist ? [1] : [])]
  const alpha = (numtaps - 1) / 2
  const h = new Float64Array(numtaps)
  for (let n = 0; n < numtaps; n++) {
    const m = n - alpha
    for (let b = 0; b < bands.length; b += 2)
      h[n] += bands[b + 1] * sinc(bands[b + 1] * m) - bands[b] * sinc(bands[b] * m)
  }
  const w = getWindow(window, numtaps).data
  for (let n = 0; n < numtaps; n++) h[n] *= w[n]
  if (scale) {
    const [left, right] = [bands[0], bands[1]]
    const f = left === 0 ? 0 : right === 1 ? 1 : 0.5 * (left + right)
    let s = 0
    for (let n = 0; n < numtaps; n++) s += h[n] * Math.cos(Math.PI * (n - alpha) * f)
    for (let n = 0; n < numtaps; n++) h[n] /= s
  }
  return fromData(h)
}

/** Kaiser's β for a stopband attenuation of A dB, as `scipy.signal.kaiser_beta` (Kaiser, 1974). */
export function kaiserBeta(attenuation: number): number {
  const a = attenuation
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/** Attenuation (dB) of a Kaiser FIR filter of `numtaps` taps and transition width (fraction of Nyquist). */
export function kaiserAttenuation(numtaps: number, width: number): number {
  return 2.285 * (numtaps - 1) * Math.PI * width + 7.95
}

/**
 * Kaiser's design formulas, as `scipy.signal.kaiserord`: the taps and β for a ripple (attenuation) of A dB and a
 * transition width given as a fraction of Nyquist.
 */
export function kaiserOrder(ripple: number, width: number): { numtaps: number; beta: number } {
  const a = Math.abs(ripple)
  if (a < 8) throw new RangeError('kaiserOrder: the attenuation must be at least 8 dB')
  return { numtaps: Math.ceil((a - 7.95) / 2.285 / (Math.PI * width) + 1), beta: kaiserBeta(a) }
}

// ── IIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Zeros, poles and gain of a filter; zeros and poles as complex vectors. */
export interface Zpk {
  zeros: ComplexTensor
  poles: ComplexTensor
  gain: number
}

/** A designed IIR filter: transfer-function coefficients and its zeros, poles and gain. */
export interface IirFilter extends Zpk {
  /** Numerator coefficients b₀, b₁, … of B(z) = Σ b_k z^{−k}. */
  b: Tensor
  /** Denominator coefficients, a₀ = 1. */
  a: Tensor
}

type Proto = { z: C[]; p: C[]; k: number }

/** Analog Butterworth prototype (cutoff 1 rad/s), as `scipy.signal.buttap`. */
function buttap(n: number): Proto {
  const p = Array.from({ length: n }, (_, i) => cscale(cexp(c(0, (Math.PI * (-n + 1 + 2 * i)) / (2 * n))), -1))
  return { z: [], p, k: 1 }
}

/** Analog Chebyshev type I prototype with passband ripple rp dB, as `scipy.signal.cheb1ap`. */
function cheb1ap(n: number, rp: number): Proto {
  const eps = Math.sqrt(10 ** (0.1 * rp) - 1)
  const mu = Math.asinh(1 / eps) / n
  const p = Array.from({ length: n }, (_, i) => {
    const theta = (Math.PI * (-n + 1 + 2 * i)) / (2 * n)
    // −sinh(μ + iθ) = −(sinh μ cos θ + i cosh μ sin θ).
    return c(-Math.sinh(mu) * Math.cos(theta), -Math.cosh(mu) * Math.sin(theta))
  })
  let k = prod(p.map((v) => cscale(v, -1))).re
  if (n % 2 === 0) k /= Math.sqrt(1 + eps * eps)
  return { z: [], p, k }
}

/** Analog Chebyshev type II prototype with stopband attenuation rs dB, as `scipy.signal.cheb2ap`. */
function cheb2ap(n: number, rs: number): Proto {
  const de = 1 / Math.sqrt(10 ** (0.1 * rs) - 1)
  const mu = Math.asinh(1 / de) / n
  const ms: number[] = []
  for (let m = -n + 1; m < n; m += 2) if (n % 2 === 0 || m !== 0) ms.push(m)
  // z = −conj(i / sin(mπ/2n)) = i / sin(mπ/2n).
  const z = ms.map((m) => c(0, 1 / Math.sin((m * Math.PI) / (2 * n))))
  const p = Array.from({ length: n }, (_, i) => {
    const q = cscale(cexp(c(0, (Math.PI * (-n + 1 + 2 * i)) / (2 * n))), -1)
    return cdiv(c(1), c(Math.sinh(mu) * q.re, Math.cosh(mu) * q.im))
  })
  const k = cdiv(prod(p.map((v) => cscale(v, -1))), prod(z.map((v) => cscale(v, -1)))).re
  return { z, p, k }
}

function lp2lp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.length - z.length
  return { z: z.map((v) => cscale(v, wo)), p: p.map((v) => cscale(v, wo)), k: k * wo ** degree }
}

function lp2hp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.length - z.length
  const gain = k * cdiv(prod(z.map((v) => cscale(v, -1))), prod(p.map((v) => cscale(v, -1)))).re
  return {
    z: [...z.map((v) => cdiv(c(wo), v)), ...Array.from({ length: degree }, () => c(0))],
    p: p.map((v) => cdiv(c(wo), v)),
    k: gain,
  }
}

function lp2bp({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.length - z.length
  const split = (roots: C[]) => {
    const lp = roots.map((v) => cscale(v, bw / 2))
    const root = lp.map((v) => csqrt(csub(cmul(v, v), c(wo * wo))))
    return [...lp.map((v, i) => cadd(v, root[i])), ...lp.map((v, i) => csub(v, root[i]))]
  }
  return { z: [...split(z), ...Array.from({ length: degree }, () => c(0))], p: split(p), k: k * bw ** degree }
}

function lp2bs({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.length - z.length
  const split = (roots: C[]) => {
    const hp = roots.map((v) => cdiv(c(bw / 2), v))
    const root = hp.map((v) => csqrt(csub(cmul(v, v), c(wo * wo))))
    return [...hp.map((v, i) => cadd(v, root[i])), ...hp.map((v, i) => csub(v, root[i]))]
  }
  const gain = k * cdiv(prod(z.map((v) => cscale(v, -1))), prod(p.map((v) => cscale(v, -1)))).re
  return {
    z: [
      ...split(z),
      ...Array.from({ length: degree }, () => c(0, wo)),
      ...Array.from({ length: degree }, () => c(0, -wo)),
    ],
    p: split(p),
    k: gain,
  }
}

/** The bilinear transform s = 2 fs (z − 1)/(z + 1), as `scipy.signal.bilinear_zpk`. */
function bilinear({ z, p, k }: Proto, fs: number): Proto {
  const degree = p.length - z.length
  const fs2 = c(2 * fs)
  return {
    z: [...z.map((v) => cdiv(cadd(fs2, v), csub(fs2, v))), ...Array.from({ length: degree }, () => c(-1))],
    p: p.map((v) => cdiv(cadd(fs2, v), csub(fs2, v))),
    k: k * cdiv(prod(z.map((v) => csub(fs2, v))), prod(p.map((v) => csub(fs2, v)))).re,
  }
}

const complexVector = (xs: C[]): ComplexTensor =>
  complexOf(
    Float64Array.from(xs, (v) => v.re),
    Float64Array.from(xs, (v) => v.im),
  )

/** Options for `iirfilter`. */
export interface IirOptions {
  btype?: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  ftype?: 'butter' | 'cheby1' | 'cheby2'
  /** Passband ripple (dB), Chebyshev I. */
  rp?: number
  /** Stopband attenuation (dB), Chebyshev II. */
  rs?: number
  fs?: number
}

/**
 * An IIR filter of order n, as `scipy.signal.iirfilter`: an analog prototype, frequency-transformed to the pre-warped
 * edges 4 tan(πWₙ/2) (fs = 2), then mapped by the bilinear transform. `wn` is one edge (low/high-pass) or two
 * (band-pass/stop). For Butterworth the edge is the −3 dB point; Chebyshev I, the passband edge; Chebyshev II, the
 * stopband edge.
 */
export function iirfilter(n: number, wn: number | readonly [number, number], options: IirOptions = {}): IirFilter {
  const { ftype = 'butter', rp = 1, rs = 40 } = options
  const btype = options.btype ?? (typeof wn === 'number' ? 'lowpass' : 'bandpass')
  const nyq = (options.fs ?? 2) / 2
  const edges = (typeof wn === 'number' ? [wn] : [...wn]).map((f) => f / nyq)
  if (edges.some((f) => !(f > 0 && f < 1)))
    throw new RangeError('iirfilter: edges must lie strictly between 0 and Nyquist')
  const two = btype === 'bandpass' || btype === 'bandstop'
  if (two !== (edges.length === 2)) throw new RangeError(`iirfilter: ${btype} needs ${two ? 'two edges' : 'one edge'}`)
  let proto = ftype === 'butter' ? buttap(n) : ftype === 'cheby1' ? cheb1ap(n, rp) : cheb2ap(n, rs)
  const warped = edges.map((f) => 4 * Math.tan((Math.PI * f) / 2))
  if (btype === 'lowpass') proto = lp2lp(proto, warped[0])
  else if (btype === 'highpass') proto = lp2hp(proto, warped[0])
  else {
    const bw = warped[1] - warped[0]
    const wo = Math.sqrt(warped[0] * warped[1])
    proto = btype === 'bandpass' ? lp2bp(proto, wo, bw) : lp2bs(proto, wo, bw)
  }
  const d = bilinear(proto, 2)
  const b = poly(d.z).map((v) => v * d.k)
  const a = poly(d.p)
  return {
    b: fromData(Float64Array.from(b)),
    a: fromData(Float64Array.from(a)),
    zeros: complexVector(d.z),
    poles: complexVector(d.p),
    gain: d.k,
  }
}

/** A Butterworth filter (maximally flat passband), as `scipy.signal.butter`. */
export function butter(
  n: number,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype'> = {},
): IirFilter {
  return iirfilter(n, wn, { ...options, ftype: 'butter' })
}

/** A Chebyshev type I filter (equiripple passband of `rp` dB), as `scipy.signal.cheby1`. */
export function cheby1(
  n: number,
  rp: number,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rp'> = {},
): IirFilter {
  return iirfilter(n, wn, { ...options, ftype: 'cheby1', rp })
}

/** A Chebyshev type II filter (equiripple stopband `rs` dB down), as `scipy.signal.cheby2`. */
export function cheby2(
  n: number,
  rs: number,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rs'> = {},
): IirFilter {
  return iirfilter(n, wn, { ...options, ftype: 'cheby2', rs })
}

// ── Filtering ─────────────────────────────────────────────────────────────────────────────────────────────────────

function normalised(bIn: Signal, aIn: Signal): { b: Float64Array; a: Float64Array; n: number } {
  const b0 = readSignal(bIn, 'filter b')
  const a0 = readSignal(aIn, 'filter a')
  if (a0.length === 0 || a0[0] === 0) throw new RangeError('filter: a[0] must be non-zero')
  const n = Math.max(b0.length, a0.length)
  const b = new Float64Array(n)
  const a = new Float64Array(n)
  for (let i = 0; i < b0.length; i++) b[i] = b0[i] / a0[0]
  for (let i = 0; i < a0.length; i++) a[i] = a0[i] / a0[0]
  return { b, a, n }
}

function lfilterRaw(
  b: Float64Array,
  a: Float64Array,
  x: Float64Array,
  zi: Float64Array,
): { y: Float64Array; zf: Float64Array } {
  const n = b.length
  const z = Float64Array.from(zi)
  const y = new Float64Array(x.length)
  for (let t = 0; t < x.length; t++) {
    const xt = x[t]
    const yt = b[0] * xt + (n > 1 ? z[0] : 0)
    for (let i = 0; i < n - 2; i++) z[i] = b[i + 1] * xt + z[i + 1] - a[i + 1] * yt
    if (n > 1) z[n - 2] = b[n - 1] * xt - a[n - 1] * yt
    y[t] = yt
  }
  return { y, zf: z }
}

/**
 * Filter a signal by the difference equation a₀y[n] = Σ b_k x[n−k] − Σ_{k≥1} a_k y[n−k], as `scipy.signal.lfilter`
 * (transposed direct form II). `zi` is the initial state (length max(|a|, |b|) − 1; default zeros); the final state is
 * returned as `zf`.
 */
export function lfilter(b: Signal, a: Signal, x: Signal, { zi }: { zi?: Signal } = {}): { y: Tensor; zf: Tensor } {
  const f = normalised(b, a)
  const z = zi ? readSignal(zi, 'lfilter zi') : new Float64Array(f.n - 1)
  if (z.length !== f.n - 1) throw new RangeError(`lfilter: zi must have length ${f.n - 1}`)
  const { y, zf } = lfilterRaw(f.b, f.a, readSignal(x, 'lfilter'), z)
  return { y: fromData(y), zf: fromData(zf) }
}

/**
 * The initial state of `lfilter` for a step response in steady state, as `scipy.signal.lfilter_zi`: solves
 * (I − Cᵀ) zi = b[1:] − a[1:] b₀ with C the companion matrix of a. Multiply by x[0] to start a signal without a
 * transient.
 */
export function lfilterZi(b: Signal, a: Signal): Tensor {
  const f = normalised(b, a)
  const m = f.n - 1
  if (m === 0) return fromData(new Float64Array(0))
  const M = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    M[i * m + i] = 1
    M[i * m] += f.a[i + 1]
    if (i + 1 < m) M[i * m + i + 1] -= 1
  }
  const rhs = Float64Array.from({ length: m }, (_, i) => f.b[i + 1] - f.a[i + 1] * f.b[0])
  return solve(fromData(M, [m, m]), fromData(rhs))
}

/**
 * Zero-phase filtering, as `scipy.signal.filtfilt` (Gustafsson, 1996, IEEE Trans. Signal Process. 44(4)): extend the
 * signal by `padlen` samples at each end (odd reflection by default), filter forwards and backwards with steady-state
 * initial conditions, and trim. The result has no phase shift and the squared magnitude response.
 */
export function filtfilt(
  b: Signal,
  a: Signal,
  x: Signal,
  options: { padtype?: 'odd' | 'even' | 'constant' | 'none'; padlen?: number } = {},
): Tensor {
  const f = normalised(b, a)
  const v = readSignal(x, 'filtfilt')
  const padtype = options.padtype ?? 'odd'
  const edge = padtype === 'none' ? 0 : (options.padlen ?? 3 * f.n)
  if (edge >= v.length) throw new RangeError(`filtfilt: the signal must be longer than padlen = ${edge}`)
  const n = v.length
  const ext = new Float64Array(n + 2 * edge)
  for (let i = 0; i < edge; i++) {
    const l = v[edge - i]
    const r = v[n - 2 - i]
    ext[i] = padtype === 'odd' ? 2 * v[0] - l : padtype === 'even' ? l : v[0]
    ext[edge + n + i] = padtype === 'odd' ? 2 * v[n - 1] - r : padtype === 'even' ? r : v[n - 1]
  }
  ext.set(v, edge)
  const zi = f.n > 1 ? (lfilterZi(f.b, f.a).data as Float64Array) : new Float64Array(0)
  const forward = lfilterRaw(
    f.b,
    f.a,
    ext,
    zi.map((z) => z * ext[0]),
  ).y
  forward.reverse()
  const backward = lfilterRaw(
    f.b,
    f.a,
    forward,
    zi.map((z) => z * forward[0]),
  ).y
  backward.reverse()
  return fromData(backward.slice(edge, edge + n))
}

// ── Responses ─────────────────────────────────────────────────────────────────────────────────────────────────────

function evaluate(coef: Float64Array, w: number): C {
  let re = 0
  let im = 0
  for (let k = 0; k < coef.length; k++) {
    re += coef[k] * Math.cos(w * k)
    im -= coef[k] * Math.sin(w * k)
  }
  return c(re, im)
}

/** Options for `freqz` and `groupDelay`. */
export interface ResponseOptions {
  /** Number of frequencies. Default 512. */
  n?: number
  /** Cover [0, 2π) instead of [0, π). Default false. */
  whole?: boolean
  /** Include the last point (π) of the half range. Default false, as scipy. */
  includeNyquist?: boolean
  /** Report frequencies in the units of fs (default: radians per sample). */
  fs?: number
}

function frequencies({ n = 512, whole = false, includeNyquist = false }: ResponseOptions): Float64Array {
  const last = whole ? 2 * Math.PI : Math.PI
  const endpoint = includeNyquist && !whole
  const div = endpoint ? n - 1 : n
  return Float64Array.from({ length: n }, (_, i) => (last * i) / div)
}

/**
 * The frequency response H(e^{iω}) = B(e^{iω}) / A(e^{iω}), as `scipy.signal.freqz`: n frequencies evenly spaced on
 * [0, π) (or [0, 2π) with `whole`). Returns the frequencies `w` (rad/sample, or in fs units) and the complex `h`.
 */
export function freqz(b: Signal, a: Signal = [1], options: ResponseOptions = {}): { w: Tensor; h: ComplexTensor } {
  const bv = readSignal(b, 'freqz b')
  const av = readSignal(a, 'freqz a')
  const w = frequencies(options)
  const re = new Float64Array(w.length)
  const im = new Float64Array(w.length)
  w.forEach((omega, i) => {
    const h = cdiv(evaluate(bv, omega), evaluate(av, omega))
    re[i] = h.re
    im[i] = h.im
  })
  const scale = options.fs === undefined ? 1 : options.fs / (2 * Math.PI)
  return { w: fromData(w.map((v) => v * scale)), h: complexOf(re, im) }
}

/**
 * Group delay τ(ω) = −dφ/dω in samples, as `scipy.signal.group_delay`: with c = b ∗ reverse(a),
 * τ = Re{Σ k c_k e^{−iωk} / Σ c_k e^{−iωk}} − (|a| − 1). Where the response is zero (the ratio is undefined) the delay
 * is NaN and `singular` counts those frequencies.
 */
export function groupDelay(
  b: Signal,
  a: Signal = [1],
  options: ResponseOptions = {},
): { w: Tensor; delay: Tensor; singular: number } {
  const bv = readSignal(b, 'groupDelay b')
  const av = readSignal(a, 'groupDelay a')
  const cc = new Float64Array(bv.length + av.length - 1)
  for (let i = 0; i < bv.length; i++) for (let j = 0; j < av.length; j++) cc[i + j] += bv[i] * av[av.length - 1 - j]
  const cr = cc.map((v, k) => v * k)
  const w = frequencies(options)
  let singular = 0
  const delay = w.map((omega) => {
    const den = evaluate(cc, omega)
    if (Math.hypot(den.re, den.im) < 10 * Number.EPSILON * cc.reduce((s, v) => s + Math.abs(v), 0)) {
      singular++
      return NaN
    }
    return cdiv(evaluate(cr, omega), den).re - (av.length - 1)
  })
  const scale = options.fs === undefined ? 1 : options.fs / (2 * Math.PI)
  return { w: fromData(w.map((v) => v * scale)), delay: fromData(delay), singular }
}

/** Unwrap a phase sequence so that consecutive values never jump by more than π, as `numpy.unwrap`. */
export function unwrap(phase: Tensor | ArrayLike<number>, { discont = Math.PI }: { discont?: number } = {}): Tensor {
  const p = readValues(phase)
  const out = Float64Array.from(p)
  let offset = 0
  for (let i = 1; i < p.length; i++) {
    const d = p[i] - p[i - 1]
    // numpy: map d into [−π, π), keeping +π for positive jumps; correct only when |d| ≥ discont.
    let dm = ((((d + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI
    if (dm === -Math.PI && d > 0) dm = Math.PI
    if (Math.abs(d) >= discont) offset += dm - d
    out[i] = p[i] + offset
  }
  return fromData(out)
}

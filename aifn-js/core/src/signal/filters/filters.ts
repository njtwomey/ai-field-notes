/**
 * Digital filters with scipy.signal's conventions: window-method FIR design (`firwin`, Kaiser's formulas), IIR design
 * from analog prototypes by the bilinear transform (`iirfilter`, `butter`, `cheby1`, `cheby2`; Oppenheim and Schafer,
 * 2010, "Discrete-Time Signal Processing", §7.1–7.3), filtering by the transposed direct form II (`lfilter`), zero-phase
 * filtering (`filtfilt`, Gustafsson's initial conditions as in scipy), and frequency and group-delay responses.
 *
 * Designs return an `LtiSystem` (discrete, dt = 1/fs; dt = 1 without `fs`), and the filtering and response
 * functions take one. Frequencies: without `fs`, cutoffs are fractions of the Nyquist frequency in (0, 1), as in
 * scipy; with `fs`, they are in the same units as `fs`.
 */

import { solve } from 'aifn/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn/foundation/tensor'
import type { LtiSystem, Scalar, Signal, Size, Spectrum, VectorLike } from 'aifn/foundation/contracts'
import { DomainError } from 'aifn/foundation/errors'
import {
  complex,
  convert,
  responseAt,
  toTransferFunction,
  transferFunction,
  zerosPolesGain,
  type Complex,
  type LtiOf,
  type TransferFunctionForm,
} from 'aifn/systems'
import { getWindow, type WindowSpec } from 'aifn/signal/windows'
import { readSamples, signal, spectrum, type SignalInput } from '../signal'

type C = Complex
const c = complex.of
const { add: cadd, sub: csub, mul: cmul, scale: cscale, div: cdiv, sqrt: csqrt, exp: cexp, product: prod } = complex

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
  /** Sampling frequency; the system gets dt = 1/fs. Default 2 (cutoffs as fractions of Nyquist, dt = 1). */
  fs?: Scalar
}

/**
 * A linear-phase FIR filter by the window method, as `scipy.signal.firwin`: the ideal band-pass impulse response
 * Σ (f_hi sinc(f_hi m) − f_lo sinc(f_lo m)) over the passbands, times a window. `cutoff` is one edge or a list of band
 * edges. An odd number of taps is required when the filter passes the Nyquist frequency. Returns the FIR system
 * b(z⁻¹)/1: its taps are `sys.repr.b`.
 */
export function firwin(
  numtaps: Size,
  cutoff: Scalar | readonly Scalar[],
  options: FirwinOptions = {},
): LtiOf<TransferFunctionForm> {
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
  return transferFunction(h, [1], { dt: dtOf(options.fs) })
}

/** The sampling interval of a design: 1/fs, or 1 when the frequencies are fractions of Nyquist. */
const dtOf = (fs: Scalar | undefined) => (fs === undefined ? 1 : 1 / fs)

/** Kaiser's β for a stopband attenuation of A dB, as `scipy.signal.kaiser_beta` (Kaiser, 1974). */
export function kaiserBeta(attenuation: Scalar): Scalar {
  const a = attenuation
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/** Attenuation (dB) of a Kaiser FIR filter of `numtaps` taps and transition width (fraction of Nyquist). */
export function kaiserAttenuation(numtaps: Size, width: Scalar): Scalar {
  return 2.285 * (numtaps - 1) * Math.PI * width + 7.95
}

/**
 * Kaiser's design formulas, as `scipy.signal.kaiserord`: the taps and β for a ripple (attenuation) of A dB and a
 * transition width given as a fraction of Nyquist.
 */
export function kaiserOrder(ripple: Scalar, width: Scalar): { numtaps: Size; beta: Scalar } {
  const a = Math.abs(ripple)
  if (a < 8) throw new RangeError('kaiserOrder: the attenuation must be at least 8 dB')
  return { numtaps: Math.ceil((a - 7.95) / 2.285 / (Math.PI * width) + 1), beta: kaiserBeta(a) }
}

// ── IIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

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

/** Options for `iirfilter`. */
export interface IirOptions {
  btype?: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  ftype?: 'butter' | 'cheby1' | 'cheby2'
  /** Passband ripple (dB), Chebyshev I. */
  rp?: number
  /** Stopband attenuation (dB), Chebyshev II. */
  rs?: number
  /** Sampling frequency; the system gets dt = 1/fs. Default 2 (edges as fractions of Nyquist, dt = 1). */
  fs?: Scalar
  /** The representation of the result: `zpk` (the design's exact form), `tf` or `sos`. Default `zpk`. */
  output?: 'zpk' | 'tf' | 'sos'
}

/**
 * An IIR filter of order n, as `scipy.signal.iirfilter`: an analog prototype, frequency-transformed to the pre-warped
 * edges 4 tan(πWₙ/2) (fs = 2), then mapped by the bilinear transform. `wn` is one edge (low/high-pass) or two
 * (band-pass/stop). For Butterworth the edge is the −3 dB point; Chebyshev I, the passband edge; Chebyshev II, the
 * stopband edge. Returns the discrete system (dt = 1/fs) in the `output` representation.
 */
export function iirfilter(n: Size, wn: Scalar | readonly [Scalar, Scalar], options: IirOptions = {}): LtiSystem {
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
  const sys = zerosPolesGain(d.z, d.p, d.k, { dt: dtOf(options.fs) })
  return convert(sys, options.output ?? 'zpk')
}

/** A Butterworth filter (maximally flat passband), as `scipy.signal.butter`. */
export function butter(
  n: Size,
  wn: Scalar | readonly [Scalar, Scalar],
  options: Omit<IirOptions, 'ftype'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'butter' })
}

/** A Chebyshev type I filter (equiripple passband of `rp` dB), as `scipy.signal.cheby1`. */
export function cheby1(
  n: number,
  rp: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rp'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby1', rp })
}

/** A Chebyshev type II filter (equiripple stopband `rs` dB down), as `scipy.signal.cheby2`. */
export function cheby2(
  n: number,
  rs: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rs'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby2', rs })
}

// ── Filtering ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The normalised coefficients (a₀ = 1) of a discrete system, padded to a common length n. */
function coefficients(sys: LtiSystem, where: string): { b: Float64Array; a: Float64Array; n: Size } {
  if (sys.domain !== 'discrete') throw new DomainError(where, `${where}: the system must be discrete`)
  const tf = toTransferFunction(sys).repr
  const b0 = dense.data(tf.b)
  const a0 = dense.data(tf.a)
  if (a0.length === 0 || a0[0] === 0) throw new DomainError(where, `${where}: a[0] must be non-zero`)
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

/** The sections of a second-order-sections system, each normalised to a₀ = 1; null for other forms. */
function sections(sys: LtiSystem): { b: Float64Array; a: Float64Array }[] | null {
  if (sys.repr.form !== 'sos') return null
  const s = dense.data(sys.repr.sections)
  return Array.from({ length: sys.repr.sections.shape[0] }, (_, k) => {
    const a0 = s[6 * k + 3]
    return {
      b: Float64Array.from(s.subarray(6 * k, 6 * k + 3), (v) => v / a0),
      a: Float64Array.from(s.subarray(6 * k + 3, 6 * k + 6), (v) => v / a0),
    }
  })
}

const output = (y: Float64Array, like: { fs: Scalar; t0: Scalar; unit?: string }): Signal =>
  signal(fromData(y, [y.length]), { fs: like.fs, t0: like.t0, unit: like.unit })

/**
 * Filters a signal through a discrete system by its difference equation a₀y[n] = Σ b_k x[n−k] − Σ_{k≥1} a_k y[n−k],
 * as `scipy.signal.lfilter` (transposed direct form II); a second-order-sections system runs as a cascade of
 * sections, as `scipy.signal.sosfilt`, which is better conditioned at high order. `zi` is the initial state (length
 * max(|a|, |b|) − 1, or 2 per section; default zeros); the final state is returned as `zf` ([sections, 2] for sos).
 * The output keeps the input's sample rate and start time.
 */
export function lfilter(sys: LtiSystem, x: SignalInput, { zi }: { zi?: VectorLike } = {}): { y: Signal; zf: Tensor } {
  if (sys.domain !== 'discrete') throw new DomainError('lfilter', 'lfilter: the system must be discrete')
  const input = readSamples(x, 'lfilter')
  const secs = sections(sys)
  if (secs) {
    const z = zi ? dense.toF64(zi, 'lfilter zi') : new Float64Array(2 * secs.length)
    if (z.length !== 2 * secs.length) throw new RangeError(`lfilter: zi must have ${2 * secs.length} values`)
    let v: Float64Array = input.values
    const zf = new Float64Array(2 * secs.length)
    secs.forEach((sec, k) => {
      const out = lfilterRaw(sec.b, sec.a, v, z.subarray(2 * k, 2 * k + 2))
      zf.set(out.zf, 2 * k)
      v = out.y
    })
    return { y: output(v, input), zf: fromData(zf, [secs.length, 2]) }
  }
  const f = coefficients(sys, 'lfilter')
  const z = zi ? dense.toF64(zi, 'lfilter zi') : new Float64Array(f.n - 1)
  if (z.length !== f.n - 1) throw new RangeError(`lfilter: zi must have length ${f.n - 1}`)
  const { y, zf } = lfilterRaw(f.b, f.a, input.values, z)
  return { y: output(y, input), zf: fromData(zf, [zf.length]) }
}

function steadyState(b: Float64Array, a: Float64Array, n: Size): Float64Array {
  const m = n - 1
  if (m === 0) return new Float64Array(0)
  const M = new Float64Array(m * m)
  for (let i = 0; i < m; i++) {
    M[i * m + i] = 1
    M[i * m] += a[i + 1]
    if (i + 1 < m) M[i * m + i + 1] -= 1
  }
  const rhs = Float64Array.from({ length: m }, (_, i) => b[i + 1] - a[i + 1] * b[0])
  return dense.data(solve(fromData(M, [m, m]), fromData(rhs, [m])))
}

/**
 * The initial state of `lfilter` for a step response in steady state, as `scipy.signal.lfilter_zi`: solves
 * (I − Cᵀ) zi = b[1:] − a[1:] b₀ with C the companion matrix of a. Multiply by x[0] to start a signal without a
 * transient.
 */
export function lfilterZi(sys: LtiSystem): Tensor {
  const f = coefficients(sys, 'lfilterZi')
  const zi = steadyState(f.b, f.a, f.n)
  return fromData(zi, [zi.length])
}

/**
 * Zero-phase filtering, as `scipy.signal.filtfilt` (Gustafsson, 1996, IEEE Trans. Signal Process. 44(4)): extend the
 * signal by `padlen` samples at each end (odd reflection by default), filter forwards and backwards with steady-state
 * initial conditions, and trim. The result has no phase shift and the squared magnitude response.
 */
export function filtfilt(
  sys: LtiSystem,
  x: SignalInput,
  options: { padtype?: 'odd' | 'even' | 'constant' | 'none'; padlen?: Size } = {},
): Signal {
  const f = coefficients(sys, 'filtfilt')
  const input = readSamples(x, 'filtfilt')
  const v = input.values
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
  const zi = steadyState(f.b, f.a, f.n)
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
  return output(backward.slice(edge, edge + n), input)
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
  n?: Size
  /** Cover [0, 2π) instead of [0, π). Default false. */
  whole?: boolean
  /** Include the last point (π) of the half range. Default false, as scipy. */
  includeNyquist?: boolean
  /**
   * The frequency axis: `hz` (in units of fs = 1/dt; cycles per sample when dt = 1) or `rad/sample` (scipy's default
   * without fs). Default `hz`.
   */
  axis?: 'hz' | 'rad/sample'
}

function frequencies({ n = 512, whole = false, includeNyquist = false }: ResponseOptions): Float64Array {
  const last = whole ? 2 * Math.PI : Math.PI
  const endpoint = includeNyquist && !whole
  const div = endpoint ? n - 1 : n
  return Float64Array.from({ length: n }, (_, i) => (last * i) / div)
}

/** The axis tag, the factor from rad/sample to it, and fs. */
function axisOf(sys: LtiSystem, options: ResponseOptions): { axis: Spectrum['axis']; scale: Scalar; fs: Scalar } {
  const fs = 1 / (sys.dt ?? 1)
  if ((options.axis ?? 'hz') === 'rad/sample') return { axis: 'rad/sample', scale: 1, fs }
  return { axis: sys.dt === 1 ? 'cycles/sample' : 'hz', scale: fs / (2 * Math.PI), fs }
}

/**
 * The frequency response H(e^{iω}) of a discrete system, as `scipy.signal.freqz`: n frequencies evenly spaced on
 * [0, π) (or [0, 2π) with `whole`), evaluated in the system's own representation (`responseAt` of `aifn/systems`).
 * Returns a `Spectrum` (`quantity: 'response'`, complex values [n, 2]) with frequencies in Hz (fs = 1/dt) or
 * rad/sample.
 */
export function freqz(sys: LtiSystem, options: ResponseOptions = {}): Spectrum {
  if (sys.domain !== 'discrete') throw new DomainError('freqz', 'freqz: the system must be discrete')
  const w = frequencies(options)
  const at = responseAt(sys)
  const { axis, scale, fs } = axisOf(sys, options)
  return spectrum({
    f: fromData(
      w.map((v) => v * scale),
      [w.length],
    ),
    axis,
    values: complex.toPairs(Array.from(w, (v) => at(v))),
    quantity: 'response',
    sided: options.whole ? 'two' : 'one',
    fs,
  })
}

/** The result of `groupDelay`. */
export interface GroupDelay {
  /** Frequencies, in the units of `axis`. */
  f: Tensor
  axis: Spectrum['axis']
  /** τ(ω) in samples; NaN where the response is zero. */
  delay: Tensor
  /** How many frequencies had an undefined delay. */
  singular: Size
}

/**
 * Group delay τ(ω) = −dφ/dω in samples of a discrete system, as `scipy.signal.group_delay`: with c = b ∗ reverse(a),
 * τ = Re{Σ k c_k e^{−iωk} / Σ c_k e^{−iωk}} − (|a| − 1), plus the system's delay. Where the response is zero (the
 * ratio is undefined) the delay is NaN and `singular` counts those frequencies.
 */
export function groupDelay(sys: LtiSystem, options: ResponseOptions = {}): GroupDelay {
  if (sys.domain !== 'discrete') throw new DomainError('groupDelay', 'groupDelay: the system must be discrete')
  const tf = toTransferFunction(sys).repr
  const bv = dense.data(tf.b)
  const av = dense.data(tf.a)
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
    return cdiv(evaluate(cr, omega), den).re - (av.length - 1) + sys.delay
  })
  const { axis, scale } = axisOf(sys, options)
  return {
    f: fromData(
      w.map((v) => v * scale),
      [w.length],
    ),
    axis,
    delay: fromData(delay, [w.length]),
    singular,
  }
}

/**
 * Unwraps a phase sequence so that consecutive values never jump by more than π, as `numpy.unwrap` (Itoh, 1982,
 * "Analysis of the phase unwrapping algorithm", Applied Optics 21(14)).
 */
export function unwrapPhase(phase: VectorLike, { discont = Math.PI }: { discont?: Scalar } = {}): Tensor {
  const p = dense.toF64(phase, 'unwrapPhase')
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
  return fromData(out, [out.length])
}

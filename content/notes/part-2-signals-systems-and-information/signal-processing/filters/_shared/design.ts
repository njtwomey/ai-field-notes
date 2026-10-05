/**
 * Filter-design helpers for the filters notes: window-method FIR design, Kaiser's formulas, IIR low-pass design from
 * analog prototypes by the bilinear transform, and group delay. Frequencies are in radians per sample, ω ∈ [0, π].
 * Results match scipy.signal (firwin, kaiserord, kaiser_beta, butter, cheby1, cheby2, group_delay).
 */
import { angle, complexAbs, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { freqz, lfilter } from 'aifn-compute/signal/filters'
import { getWindow, type WindowName } from 'aifn-compute/signal/windows'
import { transferFunction } from 'aifn-compute/systems'

type C = { re: number; im: number }
const c = (re: number, im = 0): C => ({ re, im })
const add = (a: C, b: C): C => c(a.re + b.re, a.im + b.im)
const sub = (a: C, b: C): C => c(a.re - b.re, a.im - b.im)
const mul = (a: C, b: C): C => c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
const div = (a: C, b: C): C => {
  const d = b.re * b.re + b.im * b.im
  return c((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
}
const scale = (a: C, k: number): C => c(a.re * k, a.im * k)
const expi = (t: number): C => c(Math.cos(t), Math.sin(t))

/** Polynomial coefficients (highest power first) of Π (z − r_k), real parts only (conjugate roots cancel). */
function poly(roots: C[]): number[] {
  let p: C[] = [c(1)]
  for (const r of roots) {
    const next: C[] = [...p.map((v) => ({ ...v })), c(0)]
    for (let i = 0; i < p.length; i++) next[i + 1] = sub(next[i + 1], mul(r, p[i]))
    p = next
  }
  return p.map((v) => v.re)
}

export type Zpk = { zeros: C[]; poles: C[]; gain: number }
export type Tf = { b: number[]; a: number[] }

/** Normalised analog low-pass prototypes (cutoff 1 rad/s), as scipy.signal.buttap, cheb1ap and cheb2ap. */
export function analogPrototype(family: 'butter' | 'cheby1' | 'cheby2', order: number, ripple = 1, stop = 40): Zpk {
  const n = order
  if (family === 'butter') {
    const poles = Array.from({ length: n }, (_, k) => expi((Math.PI * (2 * k + n + 1)) / (2 * n)))
    return { zeros: [], poles, gain: 1 }
  }
  if (family === 'cheby1') {
    const eps = Math.sqrt(10 ** (ripple / 10) - 1)
    const mu = Math.asinh(1 / eps) / n
    const poles = Array.from({ length: n }, (_, k) => {
      const theta = (Math.PI * (2 * k + 1)) / (2 * n)
      return c(-Math.sinh(mu) * Math.sin(theta), Math.cosh(mu) * Math.cos(theta))
    })
    let gain = poles.reduce((acc, p) => mul(acc, scale(p, -1)), c(1)).re
    if (n % 2 === 0) gain /= Math.sqrt(1 + eps * eps)
    return { zeros: [], poles, gain }
  }
  const de = 1 / Math.sqrt(10 ** (0.1 * stop) - 1)
  const mu = Math.asinh(1 / de) / n
  const ms = Array.from({ length: n }, (_, k) => -n + 1 + 2 * k)
  const zeroMs = n % 2 === 1 ? ms.filter((m) => m !== 0) : ms
  const zeros = zeroMs.map((m) => div(c(0, 1), c(Math.sin((m * Math.PI) / (2 * n)))))
  const poles = ms.map((m) => {
    const p = scale(expi((Math.PI * m) / (2 * n)), -1)
    return div(c(1), c(Math.sinh(mu) * p.re, Math.cosh(mu) * p.im))
  })
  const num = poles.reduce((acc, p) => mul(acc, scale(p, -1)), c(1))
  const den = zeros.reduce((acc, z) => mul(acc, scale(z, -1)), c(1))
  return { zeros, poles, gain: div(num, den).re }
}

/**
 * Digital low-pass IIR filter with cutoff ω_c (rad/sample): prototype → frequency scaling to the pre-warped analog
 * cutoff Ω_c = 2 tan(ω_c / 2) → bilinear transform s = 2 (z − 1)/(z + 1). For Butterworth ω_c is the −3 dB point; for
 * Chebyshev I the passband edge; for Chebyshev II the stopband edge.
 */
export function iirLowpass(
  family: 'butter' | 'cheby1' | 'cheby2',
  order: number,
  cutoff: number,
  ripple = 1,
  stop = 40,
) {
  const proto = analogPrototype(family, order, ripple, stop)
  const warped = 2 * Math.tan(cutoff / 2)
  const zeros = proto.zeros.map((z) => scale(z, warped))
  const poles = proto.poles.map((p) => scale(p, warped))
  const gain = proto.gain * warped ** (poles.length - zeros.length)
  // Bilinear transform with sample rate 1: z = (2 + s) / (2 − s); the degree difference adds zeros at z = −1.
  const two = c(2)
  const dz = [...zeros.map((z) => div(add(two, z), sub(two, z))), ...Array(poles.length - zeros.length).fill(c(-1))]
  const dp = poles.map((p) => div(add(two, p), sub(two, p)))
  const num = zeros.reduce((acc, z) => mul(acc, sub(two, z)), c(1))
  const den = poles.reduce((acc, p) => mul(acc, sub(two, p)), c(1))
  const k = gain * div(num, den).re
  const b = poly(dz).map((v) => v * k)
  const a = poly(dp)
  return { b, a, zeros: dz, poles: dp }
}

/** Window-method low-pass FIR of `taps` coefficients and cutoff ω_c, scaled to unit gain at DC (scipy.signal.firwin). */
export function firLowpass(taps: number, cutoff: number, win: WindowName | { kaiser: number } = 'hamming'): number[] {
  const m = (taps - 1) / 2
  const w = toFlat(getWindow(typeof win === 'string' ? win : { name: 'kaiser', beta: win.kaiser }, taps))
  const h = Array.from({ length: taps }, (_, n) => {
    const t = n - m
    const ideal = t === 0 ? cutoff / Math.PI : Math.sin(cutoff * t) / (Math.PI * t)
    return ideal * w[n]
  })
  const dc = h.reduce((s, v) => s + v, 0)
  return h.map((v) => v / dc)
}

/** Kaiser's β for a stopband attenuation of A dB (scipy.signal.kaiser_beta). */
export function kaiserBeta(attenuation: number): number {
  const a = attenuation
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/** Kaiser's estimate of the number of taps for attenuation A dB and transition width Δω rad/sample. */
export const kaiserTaps = (attenuation: number, width: number) => Math.ceil((attenuation - 7.95) / (2.285 * width) + 1)

/**
 * Group delay τ(ω) = −dφ/dω in samples, from τ = Re{ Σ n b[n] e^{−iωn} / B(e^{iω}) } − Re{ Σ n a[n] e^{−iωn} / A(e^{iω}) }.
 */
export function groupDelay(b: number[], a: number[] = [1], omega: number[]): number[] {
  const at = (coef: number[], w: number, weighted: boolean) =>
    coef.reduce((acc, v, n) => add(acc, scale(expi(-w * n), weighted ? n * v : v)), c(0))
  return omega.map((w) => {
    const tb = div(at(b, w, true), at(b, w, false)).re
    const ta = div(at(a, w, true), at(a, w, false)).re
    return tb - ta
  })
}

/** Decibels, 20 log₁₀ of a magnitude, floored at `floor` dB so zeros stay finite. */
export const db = (magnitude: number, floor = -200) => Math.max(floor, 20 * Math.log10(Math.max(magnitude, 1e-300)))

/** The difference equation a[0] y[n] = Σ b[k] x[n−k] − Σ_{k≥1} a[k] y[n−k], from rest (aifn's lfilter). */
export const applyFilter = (b: number[], a: number[], x: ArrayLike<number>): number[] =>
  toFlat(lfilter({ b, a }, x).y as Tensor)

/**
 * Frequency response H(e^{iω}) = B(e^{iω}) / A(e^{iω}) at `count` frequencies ω from 0 to π inclusive (aifn's freqz):
 * ω, |H| and the wrapped phase ∠H in radians.
 */
export function response(b: number[], a: number[] = [1], count = 512) {
  const h = freqz(transferFunction(b, a, { dt: 1 }), { n: count, includeNyquist: true, axis: 'rad/sample' })
  return {
    omega: toFlat(h.f),
    magnitude: toFlat(complexAbs(h.values) as Tensor),
    phase: toFlat(angle(h.values) as Tensor),
  }
}

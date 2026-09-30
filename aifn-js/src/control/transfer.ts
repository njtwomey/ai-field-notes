/**
 * Single-input single-output transfer functions G(s) = num(s)/den(s)·e^{−sτ}: conversion to and from state space,
 * frequency response, Bode data and stability margins, and loop algebra (series and unity feedback).
 */

import { eig } from 'aifn/ode'
import { findRoot, polynomialRoots } from 'aifn/solve'
import { fromData, toFlat, type Vector } from 'aifn/tensor'
import { asM, mul, sub, toT, type M } from './dense'
import { polynomialFromRoots } from './structure'
import { stateSpace, type StateSpace } from './system'
import { toF64, type VectorLike } from './vector'

/** A SISO transfer function; coefficients highest power of s first (numpy's and scipy's convention). */
export type TransferFunction = {
  /** Numerator coefficients b₀sᵐ + … + b_m. */
  num: Vector
  /** Denominator coefficients a₀sⁿ + … + aₙ, with a₀ ≠ 0. */
  den: Vector
  /** Input delay τ ≥ 0 (the factor e^{−sτ}); 0 for none. */
  delay: number
}

const stripLeading = (c: number[]) => {
  let k = 0
  while (k < c.length - 1 && c[k] === 0) k++
  return c.slice(k)
}

/**
 * A transfer function num(s)/den(s)·e^{−sτ}. Leading zeros are stripped. Improper functions (deg num > deg den) are
 * allowed for frequency responses but cannot be converted to state space.
 *
 * @example transferFunction([1], [1, 2, 1]) // 1/(s + 1)²
 */
export function transferFunction(
  num: VectorLike,
  den: VectorLike,
  { delay = 0 }: { delay?: number } = {},
): TransferFunction {
  const n = stripLeading(Array.from(toF64(num, 'transferFunction num')))
  const d = stripLeading(Array.from(toF64(den, 'transferFunction den')))
  if (d.length === 0 || d[0] === 0) throw new Error('transferFunction: the denominator must be nonzero')
  if (!(delay >= 0)) throw new Error('transferFunction: the delay must be non-negative')
  return { num: vec(n), den: vec(d), delay }
}

const vec = (v: number[]): Vector => fromData(Float64Array.from(v), [v.length])

/**
 * The controllable canonical realisation of a proper transfer function, as scipy's `tf2ss`: with the denominator
 * made monic, A has first row −[a₁ … aₙ] and ones on the subdiagonal, B = e₁, D = b₀ and C = [b₁ … bₙ] − b₀[a₁ … aₙ]
 * (numerator padded to degree n). The delay is dropped (state space has no pure delay); simulate it separately.
 */
export function tfToStateSpace(g: TransferFunction): StateSpace {
  const den = toFlat(g.den)
  const n = den.length - 1
  let num = toFlat(g.num)
  if (num.length - 1 > n) throw new Error('tfToStateSpace: the transfer function is improper')
  num = [...new Array(n + 1 - num.length).fill(0), ...num]
  const a = den.map((v) => v / den[0])
  const b = num.map((v) => v / den[0])
  if (n === 0) throw new Error('tfToStateSpace: a static gain has no states')
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === 0 ? -a[j + 1] : +(j === i - 1))),
  )
  const B = Array.from({ length: n }, (_, i) => [+(i === 0)])
  const C = [Array.from({ length: n }, (_, j) => b[j + 1] - b[0] * a[j + 1])]
  return stateSpace({ A, B, C, D: [[b[0]]] })
}

/**
 * The transfer function from input `input` to output `output` of a continuous state-space system:
 * G(s) = C(sI − A)⁻¹B + D. The denominator is det(sI − A) (from the eigenvalues of A) and the numerator
 * det(sI − A + BC) + (D − 1)det(sI − A) (the determinant identity scipy's `ss2tf` uses for one input and output).
 * No pole-zero cancellation is done, so an uncontrollable or unobservable mode stays in both.
 */
export function stateSpaceToTf(
  sys: StateSpace,
  { input = 0, output = 0 }: { input?: number; output?: number } = {},
): TransferFunction {
  const A = asM(sys.A, 'A')
  const Bfull = asM(sys.B, 'B')
  const Cfull = asM(sys.C, 'C')
  const n = sys.states
  const b: M = { d: Float64Array.from({ length: n }, (_, i) => Bfull.d[i * sys.inputs + input]), r: n, c: 1 }
  const c: M = { d: Float64Array.from({ length: n }, (_, j) => Cfull.d[output * n + j]), r: 1, c: n }
  const D = asM(sys.D, 'D').d[output * sys.inputs + input]
  const charPoly = (m: M) => {
    const e = eig(toT(m), { vectors: false })
    return toFlat(polynomialFromRoots({ real: e.real, imag: e.imag }))
  }
  const den = n === 0 ? [1] : charPoly(A)
  const other = n === 0 ? [1] : charPoly(sub(A, mul(b, c)))
  const num = other.map((v, i) => v + (D - 1) * den[i])
  // Round away the ~1e-16 noise that the eigenvalue route leaves in structurally zero coefficients.
  const scale = Math.max(...den.map(Math.abs), ...num.map(Math.abs))
  const clean = (v: number) => (Math.abs(v) < 1e-13 * scale ? 0 : v)
  return transferFunction(num.map(clean), den.map(clean))
}

/** Evaluate a polynomial (highest power first) at the complex point (re, im) by Horner's rule. */
function horner(c: ArrayLike<number>, re: number, im: number): [number, number] {
  let pr = 0
  let pi = 0
  for (let k = 0; k < c.length; k++) {
    const r = pr * re - pi * im + c[k]
    pi = pr * im + pi * re
    pr = r
  }
  return [pr, pi]
}

/** G(jω) at one frequency as [re, im]. */
function responseAt(g: TransferFunction, w: number): [number, number] {
  const [nr, ni] = horner(g.num.data as Float64Array, 0, w)
  const [dr, di] = horner(g.den.data as Float64Array, 0, w)
  const d2 = dr * dr + di * di
  let re = (nr * dr + ni * di) / d2
  let im = (ni * dr - nr * di) / d2
  if (g.delay) {
    const c = Math.cos(w * g.delay)
    const s = -Math.sin(w * g.delay)
    ;[re, im] = [re * c - im * s, re * s + im * c]
  }
  return [re, im]
}

/** The frequency response G(jω) at each frequency (rad/s): real and imaginary parts. */
export function freqResponse(g: TransferFunction, w: VectorLike): { real: Vector; imag: Vector } {
  const ws = toF64(w, 'freqResponse')
  const re = new Float64Array(ws.length)
  const im = new Float64Array(ws.length)
  ws.forEach((v, i) => ([re[i], im[i]] = responseAt(g, v)))
  return { real: fromData(re, [ws.length]), imag: fromData(im, [ws.length]) }
}

/** Poles and zeros of a transfer function (roots of den and num). */
export function polesZeros(g: TransferFunction): {
  poles: { real: Vector; imag: Vector }
  zeros: { real: Vector; imag: Vector }
} {
  const p = polynomialRoots(g.den)
  const z = polynomialRoots(g.num)
  return { poles: { real: p.real, imag: p.imag }, zeros: { real: z.real, imag: z.imag } }
}

/**
 * A logarithmic frequency grid spanning two decades either side of the poles and zeros (their nonzero moduli), or
 * 10⁻² … 10² rad/s when there are none, with `n` points (default 500).
 */
export function frequencyGrid(g: TransferFunction, n = 500): Vector {
  const { poles, zeros } = polesZeros(g)
  const mods: number[] = []
  for (const r of [poles, zeros]) {
    const re = toFlat(r.real)
    const im = toFlat(r.imag)
    re.forEach((v, i) => {
      const m = Math.hypot(v, im[i])
      if (m > 1e-12) mods.push(m)
    })
  }
  if (g.delay) mods.push(1 / g.delay)
  const lo = mods.length ? Math.log10(Math.min(...mods)) - 2 : -2
  const hi = mods.length ? Math.log10(Math.max(...mods)) + 2 : 2
  return fromData(
    Float64Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1))),
    [n],
  )
}

/** Bode data. */
export type Bode = {
  /** Frequencies (rad/s). */
  w: Vector
  /** |G(jω)|. */
  magnitude: Vector
  /** 20 log₁₀ |G(jω)|. */
  magnitudeDb: Vector
  /** Phase in degrees, unwrapped to be continuous in ω (a delay adds −ωτ exactly, not modulo 360°). */
  phase: Vector
}

/**
 * Bode data: magnitude and continuous phase over a frequency grid (default `frequencyGrid(g)`). The rational part's
 * phase starts at its principal value at the first frequency and is unwrapped between grid points (as numpy's
 * `unwrap`, like scipy's `bode`); the delay's phase −ωτ·180/π is added analytically, so a fine grid is needed only for
 * the rational part.
 */
export function bode(g: TransferFunction, w?: VectorLike): Bode {
  const ws = w === undefined ? toFlat(frequencyGrid(g)) : Array.from(toF64(w, 'bode'))
  const rational = { ...g, delay: 0 }
  const mag = new Float64Array(ws.length)
  const db = new Float64Array(ws.length)
  const ph = new Float64Array(ws.length)
  let prev = 0
  ws.forEach((v, i) => {
    const [re, im] = responseAt(rational, v)
    mag[i] = Math.hypot(re, im)
    db[i] = 20 * Math.log10(mag[i])
    let p = (Math.atan2(im, re) * 180) / Math.PI
    if (i > 0) p += 360 * Math.round((prev - p) / 360)
    prev = p
    ph[i] = p
  })
  const phase = ph.map((p, i) => p - (ws[i] * g.delay * 180) / Math.PI)
  const n = ws.length
  return {
    w: fromData(Float64Array.from(ws), [n]),
    magnitude: fromData(mag, [n]),
    magnitudeDb: fromData(db, [n]),
    phase: fromData(phase, [n]),
  }
}

/** Stability margins of an open loop L(s) under unity negative feedback. */
export type Margins = {
  /** The gain margin 1/|L(jω_pc)| at the phase crossover with the smallest margin (Infinity if phase never crosses −180°). */
  gainMargin: number
  gainMarginDb: number
  /** Phase crossover frequency ω_pc (NaN if none). */
  phaseCrossover: number
  /** The phase margin 180° + ∠L(jω_gc) at the gain crossover with the smallest margin (Infinity if |L| never crosses 1). */
  phaseMargin: number
  /** Gain crossover frequency ω_gc (NaN if none). */
  gainCrossover: number
  /** The delay margin: the extra delay that makes the loop unstable, phaseMargin·π/180/ω_gc (Infinity if none). */
  delayMargin: number
  /** Every crossing found on the grid, refined. */
  gainCrossovers: Vector
  phaseCrossovers: Vector
}

/**
 * Gain and phase margins of an open-loop transfer function L (Ogata, 2010, "Modern Control Engineering", §7-6): the
 * gain crossovers are where |L(jω)| = 1 and the phase crossovers where ∠L(jω) = −180° (mod 360°). Crossings are
 * bracketed on a fine logarithmic grid (default `frequencyGrid(L, 4000)`) and refined by Brent's method; the reported
 * margins are the smallest over all crossings, as python-control's `stability_margins` reports them. Margins are
 * meaningful when the closed loop's stability is decided by these crossings (e.g. a stable, minimum-phase L).
 */
export function margins(L: TransferFunction, w?: VectorLike): Margins {
  const grid = w === undefined ? toFlat(frequencyGrid(L, 4000)) : Array.from(toF64(w, 'margins'))
  const b = bode(L, grid)
  const mag = toFlat(b.magnitude)
  const phase = toFlat(b.phase)
  const magAt = (v: number) => Math.hypot(...responseAt(L, v))
  // The continuous phase near a grid index: principal value shifted to the nearest branch of the grid's phase.
  const phaseNear = (v: number, ref: number) => {
    const [re, im] = responseAt(L, v)
    const p = (Math.atan2(im, re) * 180) / Math.PI
    return p + 360 * Math.round((ref - p) / 360)
  }
  const gc: number[] = []
  const pc: number[] = []
  for (let i = 0; i + 1 < grid.length; i++) {
    const a = Math.log(mag[i])
    const c = Math.log(mag[i + 1])
    if (a === 0) gc.push(grid[i])
    else if (a * c < 0) {
      const r = findRoot((v) => Math.log(magAt(v)), [grid[i], grid[i + 1]], { xtol: 1e-14 })
      gc.push(r.x)
    }
    // Phase crossings of −180 + 360k: the branch k nearest the segment.
    const k = Math.round((phase[i] + 180) / 360)
    const target = -180 + 360 * k
    const f0 = phase[i] - target
    const f1 = phase[i + 1] - target
    if (f0 === 0) pc.push(grid[i])
    else if (f0 * f1 < 0) {
      const ref = phase[i]
      const r = findRoot((v) => phaseNear(v, ref) - target, [grid[i], grid[i + 1]], { xtol: 1e-14 })
      pc.push(r.x)
    }
  }
  let gainMargin = Infinity
  let phaseCrossover = NaN
  for (const v of pc) {
    const g = 1 / magAt(v)
    if (Math.abs(Math.log(g)) < Math.abs(Math.log(gainMargin)) || !Number.isFinite(gainMargin)) {
      gainMargin = g
      phaseCrossover = v
    }
  }
  let phaseMargin = Infinity
  let gainCrossover = NaN
  for (const v of gc) {
    const idx = nearest(grid, v)
    const p = phaseNear(v, phase[idx])
    // Phase margin measured to the nearest −180° branch.
    const pm = p + 180 - 360 * Math.round((p + 180) / 360)
    if (Math.abs(pm) < Math.abs(phaseMargin)) {
      phaseMargin = pm
      gainCrossover = v
    }
  }
  const delayMargin =
    Number.isFinite(phaseMargin) && phaseMargin > 0
      ? (phaseMargin * Math.PI) / 180 / gainCrossover
      : Number.isFinite(phaseMargin)
        ? 0
        : Infinity
  return {
    gainMargin,
    gainMarginDb: 20 * Math.log10(gainMargin),
    phaseCrossover,
    phaseMargin,
    gainCrossover,
    delayMargin,
    gainCrossovers: vec(gc),
    phaseCrossovers: vec(pc),
  }
}

function nearest(xs: number[], v: number): number {
  let best = 0
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - v) < Math.abs(xs[best] - v)) best = i
  return best
}

const polyMul = (a: number[], b: number[]) => {
  const out = new Array(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] += x * y)))
  return out
}
const polyAdd = (a: number[], b: number[]) => {
  const n = Math.max(a.length, b.length)
  const pa = [...new Array(n - a.length).fill(0), ...a]
  const pb = [...new Array(n - b.length).fill(0), ...b]
  return pa.map((v, i) => v + pb[i])
}

/** The series connection G₁G₂ (delays add). */
export function series(g1: TransferFunction, g2: TransferFunction): TransferFunction {
  return transferFunction(polyMul(toFlat(g1.num), toFlat(g2.num)), polyMul(toFlat(g1.den), toFlat(g2.den)), {
    delay: g1.delay + g2.delay,
  })
}

/**
 * The closed loop of L under negative feedback through H (default unity): L/(1 + LH), i.e.
 * num_L den_H / (den_L den_H + num_L num_H). A delay in the loop has no rational closed form: it throws.
 */
export function feedback(L: TransferFunction, H: TransferFunction = transferFunction([1], [1])): TransferFunction {
  if (L.delay || H.delay) throw new Error('feedback: a loop with delay is not rational; simulate it instead')
  const nl = toFlat(L.num)
  const dl = toFlat(L.den)
  const nh = toFlat(H.num)
  const dh = toFlat(H.den)
  return transferFunction(polyMul(nl, dh), polyAdd(polyMul(dl, dh), polyMul(nl, nh)))
}

/**
 * The Kepler problem for one body about a fixed mass, in units with G M = 1: the potential and its closed-form
 * gradient for `symplectic`, the conserved quantities and orbital elements of a start (position, velocity), the
 * closed-form conic, and the orbit integrated by velocity Verlet up to one period or until it leaves the view.
 */
import { hamiltonianSystem, symplectic, withEvents, type SeparableHamiltonian } from 'aifn-compute/dynamics/ode'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

export type V2 = [number, number]

/** The gravitational parameter μ = G M of the central mass. */
export const MU = 1

/** H(q, p) = |p|²/2 − μ/|q| for a unit-mass body; the force gradient is μ q/|q|³. */
export const KEPLER: SeparableHamiltonian = {
  potential: (q) => {
    const [x, y] = toFlat(q)
    return -MU / Math.hypot(x, y)
  },
  potentialGradient: (q) => {
    const [x, y] = toFlat(q)
    const r = Math.hypot(x, y)
    const r3 = r * r * r
    return [(MU * x) / r3, (MU * y) / r3]
  },
}

export type OrbitKind = 'ellipse' | 'parabola' | 'hyperbola'

/** The conserved quantities and orbital elements of a start (r, v). */
export type Elements = {
  /** Specific energy ℰ = v²/2 − μ/r. */
  energy: number
  /** Specific angular momentum h = x v_y − y v_x (the z-component). */
  angularMomentum: number
  /** The eccentricity vector (v × h)/μ − r/|r|, pointing at pericentre. */
  eccentricityVector: V2
  /** Its length e = √(1 + 2ℰh²/μ²). */
  eccentricity: number
  /** The semi-latus rectum p = h²/μ. */
  semiLatusRectum: number
  /** The semi-major axis a = −μ/(2ℰ) (negative for a hyperbola, infinite for a parabola). */
  semiMajorAxis: number
  /** The pericentre distance p/(1 + e). */
  pericentre: number
  /** The period 2π√(a³/μ) of a bound orbit; Infinity otherwise. */
  period: number
  /** The argument of pericentre: the angle of the eccentricity vector. */
  omega: number
  kind: OrbitKind
}

/** Energies within this of zero count as parabolic. */
const PARABOLIC = 1e-3

export function elements([x, y]: V2, [vx, vy]: V2): Elements {
  const r = Math.hypot(x, y)
  const energy = (vx * vx + vy * vy) / 2 - MU / r
  const h = x * vy - y * vx
  const ex = (vy * h) / MU - x / r
  const ey = (-vx * h) / MU - y / r
  const e = Math.hypot(ex, ey)
  const p = (h * h) / MU
  const a = -MU / (2 * energy)
  const kind: OrbitKind = Math.abs(energy) < PARABOLIC ? 'parabola' : energy < 0 ? 'ellipse' : 'hyperbola'
  return {
    energy,
    angularMomentum: h,
    eccentricityVector: [ex, ey],
    eccentricity: e,
    semiLatusRectum: p,
    semiMajorAxis: a,
    pericentre: p / (1 + e),
    period: energy < 0 ? 2 * Math.PI * Math.sqrt((a * a * a) / MU) : Infinity,
    omega: Math.atan2(ey, ex),
    kind,
  }
}

/**
 * The closed-form conic r(ν) = p/(1 + e cos ν), rotated by ω, as x and y arrays: the whole ellipse, or the branch of a
 * parabola or hyperbola out to radius `rMax`.
 */
export function conic(el: Elements, rMax: number, n = 400): { x: number[]; y: number[] } {
  const { semiLatusRectum: p, eccentricity: e, omega } = el
  // The true anomaly where the branch reaches rMax (or ±π for an ellipse that stays inside).
  const c = (p / rMax - 1) / Math.max(e, 1e-12)
  const nuMax = e < 1 && c <= -1 ? Math.PI : Math.acos(Math.max(-1, Math.min(1, c)))
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= n; i++) {
    const nu = -nuMax + (2 * nuMax * i) / n
    const r = p / (1 + e * Math.cos(nu))
    if (!(r > 0) || r > rMax * 1.001) {
      x.push(NaN)
      y.push(NaN)
      continue
    }
    x.push(r * Math.cos(nu + omega))
    y.push(r * Math.sin(nu + omega))
  }
  return { x, y }
}

export type Orbit = { t: number[]; x: number[]; y: number[]; energyError: number[]; stepSize: number; steps: number }

/** The most steps one orbit takes, whatever the start. */
export const MAX_STEPS = 20_000

/**
 * The orbit from (r0, v0) by velocity Verlet: one period of a bound orbit (at most `tMax`), or until the body passes
 * `rExit` moving outwards. The step is 2% of the dynamical time √(r_p³/μ) at pericentre, at most 0.01, and at least
 * the length of the run over `MAX_STEPS`.
 */
export function integrateOrbit(r0: V2, v0: V2, { rExit = 8, tMax = 60 } = {}): Orbit {
  const el = elements(r0, v0)
  const tEnd = Math.min(el.period, tMax)
  const rp = Math.max(el.pericentre, 1e-4)
  const stepSize = Math.max(Math.min(0.02 * Math.sqrt((rp * rp * rp) / MU), 0.01), tEnd / MAX_STEPS)
  const { rhs } = hamiltonianSystem(KEPLER)
  // Stop when the body passes rExit outwards: g = |q| − rExit rising through zero.
  const exit = {
    name: 'leaves the view',
    g: (_t: number, s: Tensor) => {
      const [x, y] = toFlat(s)
      return Math.hypot(x, y) - rExit
    },
    direction: 1 as const,
    terminal: true,
  }
  const alg = withEvents(symplectic(KEPLER, 'velocity-verlet', { stepSize, tEnd }), rhs, [exit])
  const tr = trace(alg, { q0: r0, p0: v0 }, MAX_STEPS + 1, {
    keep: 'none',
    stopOnNonFinite: true,
    record: {
      t: (s) => s.time,
      x: (s) => toFlat(s.x)[0],
      y: (s) => toFlat(s.x)[1],
      e: (s) => s.energyError,
    },
  })
  return {
    t: toFlat(tr.series.t),
    x: toFlat(tr.series.x),
    y: toFlat(tr.series.y),
    energyError: toFlat(tr.series.e),
    stepSize,
    steps: tr.meta.steps,
  }
}

/** The index of the last entry of an ascending `ts` at or before `t` (0 if none). */
export function upTo(ts: readonly number[], t: number): number {
  let lo = 0
  let hi = ts.length - 1
  if (hi < 0 || ts[0] > t) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** The area of a closed polygon by the shoelace formula (positive anticlockwise). */
export function polygonArea(x: readonly number[], y: readonly number[]): number {
  let s = 0
  for (let i = 0, j = x.length - 1; i < x.length; j = i++) s += x[j] * y[i] - x[i] * y[j]
  return s / 2
}

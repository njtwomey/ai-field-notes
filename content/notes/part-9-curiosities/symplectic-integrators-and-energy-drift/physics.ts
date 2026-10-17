/**
 * The two Hamiltonian systems of this note, written as `SeparableHamiltonian`s with closed-form gradients, and the
 * runs its figures draw. The integration is the engine's: `symplectic` and `rungeKutta` from `aifn-compute/dynamics/ode`.
 *
 * - The Kepler problem, H = |p|²/2 − 1/|q| (units with G M = 1), started at pericentre of an orbit with semi-major axis
 *   1, so the period is 2π.
 * - The pendulum, H = p²/2 − cos q, many copies at once.
 */
import {
  hamiltonianSystem,
  rungeKutta,
  symplectic,
  type SeparableHamiltonian,
  type SymplecticMethod,
} from 'aifn-compute/dynamics/ode'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

export const KEPLER: SeparableHamiltonian = {
  potential: (q) => {
    const [x, y] = toFlat(q)
    return -1 / Math.hypot(x, y)
  },
  potentialGradient: (q) => {
    const [x, y] = toFlat(q)
    const r = Math.hypot(x, y)
    const r3 = r * r * r
    return [x / r3, y / r3]
  },
}

/** The pericentre start of a Kepler orbit with a = 1 and eccentricity e: q = (1 − e, 0), p = (0, √((1 + e)/(1 − e))). */
export function pericentreStart(e: number) {
  return { q0: [1 - e, 0], p0: [0, Math.sqrt((1 + e) / (1 - e))] }
}

/** The exact orbit: the ellipse with a = 1, eccentricity e and the mass at the origin, pericentre on the +x axis. */
export function exactEllipse(e: number, n = 240) {
  const b = Math.sqrt(1 - e * e)
  const s = Array.from({ length: n + 1 }, (_, i) => (2 * Math.PI * i) / n)
  return { x: s.map((E) => Math.cos(E) - e), y: s.map((E) => b * Math.sin(E)) }
}

export type KeplerRun = {
  /** Positions at every step. */
  x: number[]
  y: number[]
  /** H − H₀ and h − h₀ (angular momentum) at every step. */
  dE: number[]
  dL: number[]
  /** The argument of pericentre (the angle of the eccentricity vector), unwrapped, in degrees, at every step. */
  omega: number[]
  /** The pericentre point r_p (cos ω, sin ω) at the end of each whole period. */
  peri: { x: number[]; y: number[] }
}

/** Runs longer than this many steps are cut short (the figure says so). */
export const MAX_STEPS = 200_000

/**
 * The diagnostics of a run from its states: energy and angular-momentum errors, and the orbit's pericentre from the
 * eccentricity vector e = (v_y h − x/r, −v_x h − y/r), sampled every `per` steps.
 */
function diagnose(s: Float64Array | number[], H0: number, L0: number, per: number): KeplerRun {
  const run: KeplerRun = { x: [], y: [], dE: [], dL: [], omega: [], peri: { x: [], y: [] } }
  let prev = 0
  let turns = 0
  for (let i = 0; 4 * i < s.length; i++) {
    const [x, y, vx, vy] = [s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]]
    const r = Math.hypot(x, y)
    const L = x * vy - y * vx
    const ex = vy * L - x / r
    const ey = -vx * L - y / r
    const w = Math.atan2(ey, ex)
    // Unwrap: the eccentricity vector turns slowly, so a jump of more than π is a crossing of ±π.
    if (i > 0) turns += Math.round((prev - w) / (2 * Math.PI))
    prev = w
    run.x.push(x)
    run.y.push(y)
    run.dE.push((vx * vx + vy * vy) / 2 - 1 / r - H0)
    run.dL.push(L - L0)
    run.omega.push(((w + 2 * Math.PI * turns) * 180) / Math.PI)
    if (i > 0 && i % per === 0) {
      const rp = (L * L) / (1 + Math.hypot(ex, ey))
      run.peri.x.push(rp * Math.cos(w))
      run.peri.y.push(rp * Math.sin(w))
    }
  }
  return run
}

/**
 * The Kepler orbit from pericentre for `periods` periods with step h, by RK4 on the first-order system and by a
 * symplectic method, recording every step.
 */
export function keplerRuns(e: number, h: number, periods: number, method: SymplecticMethod) {
  const { q0, p0 } = pericentreStart(e)
  const steps = Math.min(MAX_STEPS, Math.round((periods * 2 * Math.PI) / h))
  const per = Math.max(1, Math.round((2 * Math.PI) / h))
  const H0 = (p0[1] * p0[1]) / 2 - 1 / q0[0]
  const L0 = q0[0] * p0[1]
  const { rhs } = hamiltonianSystem(KEPLER)
  const rk = trace(rungeKutta(rhs, 'rk4', { stepSize: h }), { x0: [...q0, ...p0] }, steps, {
    keep: 'none',
    stopOnNonFinite: false,
    record: { s: (st) => toFlat(st.x) },
  })
  const sy = trace(symplectic(KEPLER, method, { stepSize: h }), { q0, p0 }, steps, {
    keep: 'none',
    stopOnNonFinite: false,
    record: { s: (st) => [...toFlat(st.q), ...toFlat(st.p)] },
  })
  return {
    steps,
    per,
    rk4: diagnose(toFlat(rk.series.s), H0, L0, per),
    sympl: diagnose(toFlat(sy.series.s), H0, L0, per),
  }
}

/**
 * A curve of `v` thinned for drawing: in each of `buckets` windows, the smallest and largest value, in step order, so
 * the band of an oscillating error survives the thinning.
 */
export function envelope(v: readonly number[], dt: number, buckets = 1200): { t: number[]; y: number[] } {
  const per = Math.max(1, Math.ceil(v.length / buckets))
  const t: number[] = []
  const y: number[] = []
  for (let s = 0; s < v.length; s += per) {
    let lo = s
    let hi = s
    for (let i = s; i < Math.min(v.length, s + per); i++) {
      if (v[i] < v[lo]) lo = i
      if (v[i] > v[hi]) hi = i
    }
    for (const i of lo <= hi ? [lo, hi] : [hi, lo]) {
      t.push(i * dt)
      y.push(v[i])
    }
  }
  return { t, y }
}

export type AreaMethod = 'euler' | 'rk4' | 'leapfrog'

/**
 * Many uncoupled pendulums as one separable system: H = Σⱼ pⱼ²/2 − cos qⱼ, so a whole set of starts is integrated by
 * one engine run.
 */
export const PENDULUMS: SeparableHamiltonian = {
  potential: (q) => -toFlat(q).reduce((s, v) => s + Math.cos(v), 0),
  potentialGradient: (q) => toFlat(q).map(Math.sin),
}

/**
 * The pendulum flow of a set of starts for `steps` steps of size h: `out[k][j]` is start j after k steps, as [q, p].
 */
export function flowSet(method: AreaMethod, starts: readonly (readonly [number, number])[], h: number, steps: number) {
  const n = starts.length
  const q0 = starts.map((s) => s[0])
  const p0 = starts.map((s) => s[1])
  let q: number[]
  let p: number[]
  if (method === 'leapfrog') {
    const tr = trace(symplectic(PENDULUMS, 'leapfrog', { stepSize: h }), { q0, p0 }, steps, {
      keep: 'none',
      stopOnNonFinite: false,
      record: { q: (s) => toFlat(s.q), p: (s) => toFlat(s.p) },
    })
    q = toFlat(tr.series.q)
    p = toFlat(tr.series.p)
  } else {
    const { rhs } = hamiltonianSystem(PENDULUMS)
    const tr = trace(rungeKutta(rhs, method, { stepSize: h }), { x0: [...q0, ...p0] }, steps, {
      keep: 'none',
      stopOnNonFinite: false,
      record: { q: (s) => toFlat(s.x).slice(0, n), p: (s) => toFlat(s.x).slice(n) },
    })
    q = toFlat(tr.series.q)
    p = toFlat(tr.series.p)
  }
  const out: [number, number][][] = []
  for (let k = 0; k * n < q.length; k++) out.push(Array.from({ length: n }, (_, j) => [q[k * n + j], p[k * n + j]]))
  return out
}

/** The area of a closed polygon by the shoelace formula (positive anticlockwise). */
export function polygonArea(pts: readonly (readonly [number, number])[]): number {
  let s = 0
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) s += pts[j][0] * pts[i][1] - pts[i][0] * pts[j][1]
  return s / 2
}

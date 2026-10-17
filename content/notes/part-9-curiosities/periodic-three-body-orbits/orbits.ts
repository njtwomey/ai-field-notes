/**
 * Three periodic solutions of the equal-mass three-body problem (m₁ = m₂ = m₃ = 1, G = 1) and a leapfrog run of a
 * kicked copy of each.
 *
 * - Euler's collinear solution: the masses at x = −1, 0, 1 rotate rigidly about the middle one. The outer masses feel
 *   1/1² + 1/2² = 5/4, so ω² = 5/4 and each outer mass moves at speed √5/2; period 4π/√5.
 * - Lagrange's equilateral solution: the masses at the corners of a unit equilateral triangle rotate rigidly. Each
 *   feels √3 toward the centre at distance 1/√3, so ω² = 3 and each moves at speed 1; period 2π/√3.
 * - The figure-eight: the initial conditions computed by Simó and printed by Chenciner and Montgomery (2000, Fig. 1):
 *   x₁ = −x₂ = (0.97000436, −0.24308753), x₃ = 0, v₃ = −2v₁ = −2v₂ = (−0.93240737, −0.86473146), period 6.32591398.
 *
 * Plain physics, no React. The integration is the engine's `symplectic` (leapfrog).
 */
import { symplectic } from 'aifn-compute/dynamics/ode'
import { live } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { angularMomentum, nBodyHamiltonian, toCentreOfMassFrame } from '../_shared/gravity'

export const MASSES = [1, 1, 1]

export type PeriodicOrbit = {
  label: string
  /** Positions (x₁, y₁, x₂, y₂, x₃, y₃) at t = 0. */
  q: readonly number[]
  /** Velocities at t = 0 (equal to the momenta, since every mass is 1). */
  v: readonly number[]
  /** The period. */
  period: number
}

const S3 = Math.sqrt(3)
const S5 = Math.sqrt(5)

export const ORBITS = {
  eight: {
    label: 'figure-eight (Moore; Chenciner and Montgomery)',
    q: [0.97000436, -0.24308753, -0.97000436, 0.24308753, 0, 0],
    v: [0.466203685, 0.43236573, 0.466203685, 0.43236573, -0.93240737, -0.86473146],
    period: 6.32591398,
  },
  lagrange: {
    label: "Lagrange's equilateral triangle",
    q: [1 / S3, 0, -1 / (2 * S3), 0.5, -1 / (2 * S3), -0.5],
    v: [0, 1, -S3 / 2, -0.5, S3 / 2, -0.5],
    period: (2 * Math.PI) / S3,
  },
  euler: {
    label: "Euler's collinear line",
    q: [-1, 0, 0, 0, 1, 0],
    v: [0, -S5 / 2, 0, 0, 0, S5 / 2],
    period: (4 * Math.PI) / S5,
  },
} satisfies Record<string, PeriodicOrbit>
export type OrbitName = keyof typeof ORBITS

export type OrbitRun = {
  /** Time of each kept frame, in periods. */
  periods: number[]
  /** Positions at each kept frame. */
  q: number[][]
  /** |E/E₀ − 1| at each kept frame. */
  energyError: number[]
  /** The angular momentum about the centre of mass (constant up to rounding). */
  angularMomentum: number
  /** The unperturbed orbit over one period, every body's path concatenated, as points for drawing. */
  reference: { x: number[]; y: number[] }[]
  /**
   * How far the triangle's shape is from any shape the unperturbed orbit passes through: the distance from the mutual
   * distances (r₁₂, r₁₃, r₂₃) to the nearest point of the unperturbed orbit's curve of them, at each kept frame. Mutual
   * distances ignore where the triangle is and how it is turned, so a slow rotation of the whole orbit does not count.
   */
  shapeDistance: number[]
}

/** The mutual distances (r₁₂, r₁₃, r₂₃) of a configuration. */
export function mutualDistances(q: ArrayLike<number>): [number, number, number] {
  const d = (i: number, j: number) => Math.hypot(q[2 * i] - q[2 * j], q[2 * i + 1] - q[2 * j + 1])
  return [d(0, 1), d(0, 2), d(1, 2)]
}

/** Every body's path over `steps` leapfrog steps of size h from (q, p), keeping every `stride`-th step. */
function integrate(q0: ArrayLike<number>, p0: ArrayLike<number>, h: number, steps: number, stride: number) {
  const solver = symplectic(nBodyHamiltonian(MASSES), 'leapfrog', { stepSize: h })
  const q: number[][] = []
  const err: number[] = []
  let e0 = NaN
  for (const { step, state } of live(solver, { q0: q0, p0: p0 })) {
    if (step === 0) e0 = state.energy
    if (step % stride === 0 || step === steps) {
      q.push(toFlat(state.q))
      err.push(Math.abs(state.energyError / e0))
    }
    if (step >= steps || state.diverged) break
  }
  return { q, err }
}

/**
 * Run an orbit for `periods` periods with `stepsPerPeriod` leapfrog steps per period, after kicking body 1's
 * x-velocity by `kick` and returning to the centre-of-mass frame.
 */
export function runOrbit(name: OrbitName, kick: number, periods: number, stepsPerPeriod: number): OrbitRun {
  const o: PeriodicOrbit = ORBITS[name]
  const h = o.period / stepsPerPeriod
  const stride = Math.max(1, Math.round(stepsPerPeriod / 100))

  // The unperturbed orbit over one period, thinned to at most about 600 points per body.
  const refStride = Math.max(1, Math.round(stepsPerPeriod / 600))
  const ref = integrate(o.q, o.v, h, stepsPerPeriod, refStride).q
  const reference = [0, 1, 2].map((b) => ({ x: ref.map((s) => s[2 * b]), y: ref.map((s) => s[2 * b + 1]) }))
  const refShapes = ref.map(mutualDistances)

  const kicked = [...o.v]
  kicked[0] += kick
  const start = toCentreOfMassFrame(MASSES, o.q, kicked)
  const run = integrate(start.q, start.p, h, periods * stepsPerPeriod, stride)

  const shapeDistance = run.q.map((s) => {
    const d = mutualDistances(s)
    let best = Infinity
    for (const r of refShapes) best = Math.min(best, Math.hypot(d[0] - r[0], d[1] - r[1], d[2] - r[2]))
    return best
  })

  return {
    periods: run.q.map((_, k) => Math.min(periods, (k * stride) / stepsPerPeriod)),
    q: run.q,
    energyError: run.err,
    angularMomentum: angularMomentum(MASSES, start.q, start.p),
    reference,
    shapeDistance,
  }
}

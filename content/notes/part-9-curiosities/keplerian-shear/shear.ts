/**
 * The physics of Keplerian shear, in units with G = M = 1 and the reference radius R₀ = 1.
 *
 * - A power-law rotation law Ω(r) = r^(−q): q = 3/2 is Kepler's, q = 1 a flat rotation curve, q = 0 rigid rotation.
 * - Tracers on a radial line wound up by the shear, in closed form.
 * - A particle nudged off a circular orbit, integrated with the engine's `rungeKutta` (with `withEvents` for the
 *   pericentres), and Hill's linearised solution in the frame rotating with the unperturbed orbit.
 */
import { rungeKutta, withEvents, type EventState, type OdeState, type Rhs } from 'aifn-compute/dynamics/ode'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

/** The angular velocity Ω(r) = r^(−q). */
export const omega = (r: number, q: number) => r ** -q

/** The epicyclic frequency of the power law, κ = √(2(2 − q)) Ω, from κ² = (1/r³) d(r⁴Ω²)/dr. */
export const epicyclicFrequency = (r: number, q: number) => Math.sqrt(2 * (2 - q)) * omega(r, q)

/** The tracers that start on the line φ = 0 at radii `r`, at time t: φ = Ω(r) t. */
export function windTracers(r: readonly number[], q: number, t: number) {
  const x = r.map((ri) => ri * Math.cos(omega(ri, q) * t))
  const y = r.map((ri) => ri * Math.sin(omega(ri, q) * t))
  return { x, y }
}

/** The pitch angle of the wound line at radius r: tan p = 1/(q Ω(r) t), in degrees (90 at t = 0). */
export const pitchAngle = (r: number, q: number, t: number) =>
  t === 0 || q === 0 ? 90 : (Math.atan(1 / (q * omega(r, q) * t)) * 180) / Math.PI

export type Potential = 'kepler' | 'flat'

/** The shear exponent q of each potential: Φ = −1/r (Kepler) or Φ = ln r (flat rotation curve, v_c = 1). */
export const SHEAR_EXPONENT: Record<Potential, number> = { kepler: 1.5, flat: 1 }

/**
 * The equations of motion in the plane with the azimuth carried along, state (x, y, vₓ, v_y, φ): the acceleration is
 * −r/r³ (Kepler) or −r/r² (flat), and φ' = (x v_y − y vₓ)/r², so φ is unwrapped.
 */
export function planarRhs(potential: Potential): Rhs {
  const power = potential === 'kepler' ? 3 : 2
  return (_t, state) => {
    const [x, y, vx, vy] = toFlat(state)
    const r2 = x * x + y * y
    const s = r2 ** (-power / 2)
    return [vx, vy, -x * s, -y * s, (x * vy - y * vx) / r2]
  }
}

export type EpicycleRun = {
  t: number[]
  /** Inertial position. */
  x: number[]
  y: number[]
  /** Radial offset ξ = r − 1 and azimuthal offset η = φ − Ω₀t (arc length at R₀ = 1) in the rotating frame. */
  xi: number[]
  eta: number[]
  /** Hill's linear prediction of ξ and η. */
  hillXi: number[]
  hillEta: number[]
  /** The pericentres found: times and azimuths. */
  pericentres: { time: number; phi: number }[]
  /** Specific energy and angular momentum at each kept step. */
  energy: number[]
  angularMomentum: number[]
}

/**
 * A particle at (1, 0) given the velocity of the circular orbit plus a kick (u₀ radial, v₀ azimuthal), integrated with
 * RK4 for `orbits` periods of the circular orbit at `stepsPerOrbit` steps each.
 */
export function epicycle(
  potential: Potential,
  u0: number,
  v0: number,
  orbits: number,
  stepsPerOrbit: number,
): EpicycleRun {
  const f = planarRhs(potential)
  const q = SHEAR_EXPONENT[potential]
  const Omega = 1 // Ω(1) = 1 for both potentials.
  const kappa = epicyclicFrequency(1, q)
  const period = 2 * Math.PI
  const h = period / stepsPerOrbit
  const tEnd = orbits * period
  const pericentre = {
    name: 'pericentre',
    g: (_t: number, s: Parameters<Rhs>[1]) => {
      const [x, y, vx, vy] = toFlat(s)
      return x * vx + y * vy
    },
    direction: 1 as const,
  }
  const alg = withEvents(rungeKutta(f, 'rk4', { stepSize: h, tEnd }), f, [pericentre])
  const tr = trace(alg, { x0: [1, 0, u0, Omega + v0, 0] }, orbits * stepsPerOrbit + 2, {
    record: { s: (st: OdeState) => toFlat(st.x) },
  })
  const steps = tr.steps as EventState<OdeState>[]
  const final = steps[steps.length - 1]
  const out: EpicycleRun = {
    t: [],
    x: [],
    y: [],
    xi: [],
    eta: [],
    hillXi: [],
    hillEta: [],
    pericentres: final.events.map((e) => ({ time: e.time, phi: toFlat(e.x)[4] })),
    energy: [],
    angularMomentum: [],
  }
  const k2 = kappa * kappa
  for (const st of steps) {
    const [x, y, vx, vy, phi] = toFlat(st.x)
    const t = st.time
    const r = Math.hypot(x, y)
    out.t.push(t)
    out.x.push(x)
    out.y.push(y)
    out.xi.push(r - 1)
    out.eta.push(phi - Omega * t)
    // Hill's equations ξ'' − 2Ωη' = 2qΩ²ξ, η'' + 2Ωξ' = 0 from ξ = η = 0 with velocities (u₀, v₀).
    const c = Math.cos(kappa * t)
    const s = Math.sin(kappa * t)
    out.hillXi.push(((2 * Omega * v0) / k2) * (1 - c) + (u0 / kappa) * s)
    out.hillEta.push(
      (v0 - (4 * Omega * Omega * v0) / k2) * t +
        ((4 * Omega * Omega * v0) / (k2 * kappa)) * s -
        ((2 * Omega * u0) / k2) * (1 - c),
    )
    const speed2 = vx * vx + vy * vy
    out.energy.push(0.5 * speed2 + (potential === 'kepler' ? -1 / r : Math.log(r)))
    out.angularMomentum.push(x * vy - y * vx)
  }
  return out
}

/**
 * Balbus and Hawley's spring model of the magnetorotational instability (Balbus & Hawley 1998, eqs. 106–114):
 * two equal masses on neighbouring orbits of a Keplerian disc, joined by a spring, in Hill's frame. The frame rotates
 * with angular velocity Ω about the central mass at the reference radius R₀; x points radially outward and y along the
 * orbital motion. Each mass obeys Hill's equations plus the spring force,
 *
 *   ẍ = 2Ωẏ + 3Ω²x + Fx,   ÿ = −2Ωẋ + Fy,
 *
 * where 3Ω² = −dΩ²/d ln R is the Keplerian tidal term. The spring pulls each mass towards the other with acceleration
 * (k/2) × separation, so the separation ξ = ξ₂ − ξ₁ feels −kξ: k (units Ω²) is the squared frequency at which the
 * pair would oscillate without rotation.
 */
import type { Rhs } from 'aifn-compute/dynamics/ode'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'

/** The state layout: positions of mass 1 and mass 2, then their velocities. */
export type PairState = [x1: number, y1: number, x2: number, y2: number, u1: number, v1: number, u2: number, v2: number]

/** The right-hand side f(t, ξ) of the two-mass system with spring constant `k` and angular velocity `omega`. */
export function pairRhs(k: number, omega = 1): Rhs {
  const tidal = 3 * omega * omega
  return (_t, state) => {
    const [x1, y1, x2, y2, u1, v1, u2, v2] = toFlat(state)
    const fx = (k / 2) * (x2 - x1)
    const fy = (k / 2) * (y2 - y1)
    return [
      u1,
      v1,
      u2,
      v2,
      2 * omega * v1 + tidal * x1 + fx,
      -2 * omega * u1 + fy,
      2 * omega * v2 + tidal * x2 - fx,
      -2 * omega * u2 - fy,
    ]
  }
}

/**
 * Masses at x = ∓δ/2 on the radial line through the origin, each moving with the local shear flow ẏ = −(3/2)Ωx: on
 * their own (k = 0) they would stay on circular orbits and the separation would only grow linearly in y.
 */
export function initialPair(delta: number, omega = 1): PairState {
  const x1 = -delta / 2
  const x2 = delta / 2
  return [x1, 0, x2, 0, 0, -1.5 * omega * x1, 0, -1.5 * omega * x2]
}

/** The distance between the two masses. */
export const separation = (s: ArrayLike<number>) => Math.hypot(s[2] - s[0], s[3] - s[1])

/**
 * The roots S = s² of the dispersion relation s⁴ + s²(κ² + 2k) + k(k − 3Ω²) = 0 for perturbations ∝ e^{st}, with
 * κ² = Ω² (Keplerian). A positive root is an exponentially growing mode.
 */
export function dispersionRoots(k: number, omega = 1): [number, number] {
  const b = omega * omega + 2 * k
  const c = k * (k - 3 * omega * omega)
  const disc = Math.sqrt(b * b - 4 * c)
  return [(-b + disc) / 2, (-b - disc) / 2]
}

/** The growth rate s of the unstable mode (0 when every mode oscillates): positive for 0 < k < 3Ω². */
export const growthRate = (k: number, omega = 1) => Math.sqrt(Math.max(0, dispersionRoots(k, omega)[0]))

/** The largest growth rate over k, (3/4)Ω, reached at k = (15/16)Ω² (Balbus & Hawley 1998, eq. 114). */
export const MAX_GROWTH = { rate: 0.75, k: 15 / 16 }

/**
 * The growth rate measured from a run: the least-squares slope of ln(separation) against time over the last half of
 * the run, by which point the growing mode dominates the oscillating one.
 */
export function measuredRate(time: ArrayLike<number>, sep: ArrayLike<number>): number {
  const rows: number[][] = []
  const logs: number[] = []
  for (let i = Math.floor(time.length / 2); i < time.length; i++) {
    const l = Math.log(sep[i])
    if (!Number.isFinite(l)) continue
    rows.push([1, time[i]])
    logs.push(l)
  }
  if (rows.length < 2) return NaN
  return toFlat(lstsq(tensor(rows), tensor(logs)).x)[1]
}

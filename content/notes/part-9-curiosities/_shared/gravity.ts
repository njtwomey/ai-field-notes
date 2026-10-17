/**
 * Newtonian gravity for N point masses in the plane, in units where the gravitational constant G = 1.
 *
 * Layout: positions q = (x_0, y_0, x_1, y_1, …) and momenta p = (m_0 vx_0, m_0 vy_0, …), each of length 2N. The
 * Hamiltonian is separable, H(q, p) = T(p) + V(q), with
 *   T(p) = Σ_i |p_i|² / (2 m_i)
 *   V(q) = −Σ_{i<j} m_i m_j / sqrt(|q_i − q_j|² + ε²),
 * where ε ≥ 0 is the softening length (ε = 0 is exact Newtonian gravity). Both gradients are in closed form, so the
 * engine's `symplectic` and `hamiltonianSystem` never trace these functions.
 *
 * Plain physics, no React: the notes three-body-problem and periodic-three-body-orbits share it.
 */
import type { SeparableHamiltonian } from 'aifn-compute/dynamics/ode'
import { toFlat } from 'aifn-compute/foundation/tensor'

type Vec = ArrayLike<number>

/** V(q) = −Σ_{i<j} m_i m_j / sqrt(r_ij² + ε²). */
export function potentialEnergy(masses: readonly number[], q: Vec, softening = 0): number {
  const n = masses.length
  const e2 = softening * softening
  let v = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = q[2 * i] - q[2 * j]
      const dy = q[2 * i + 1] - q[2 * j + 1]
      v -= (masses[i] * masses[j]) / Math.sqrt(dx * dx + dy * dy + e2)
    }
  return v
}

/** ∇V(q): ∂V/∂q_i = Σ_{j≠i} m_i m_j (q_i − q_j) / (r_ij² + ε²)^{3/2}. Its negative is the force on body i. */
export function potentialGradient(masses: readonly number[], q: Vec, softening = 0): Float64Array {
  const n = masses.length
  const e2 = softening * softening
  const g = new Float64Array(2 * n)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const dx = q[2 * i] - q[2 * j]
      const dy = q[2 * i + 1] - q[2 * j + 1]
      const r2 = dx * dx + dy * dy + e2
      const c = (masses[i] * masses[j]) / (r2 * Math.sqrt(r2))
      g[2 * i] += c * dx
      g[2 * i + 1] += c * dy
      g[2 * j] -= c * dx
      g[2 * j + 1] -= c * dy
    }
  return g
}

/** T(p) = Σ_i |p_i|² / (2 m_i). */
export function kineticEnergy(masses: readonly number[], p: Vec): number {
  let t = 0
  for (let i = 0; i < masses.length; i++) t += (p[2 * i] ** 2 + p[2 * i + 1] ** 2) / (2 * masses[i])
  return t
}

/** ∇T(p) = the velocities, p_i / m_i. */
export function velocities(masses: readonly number[], p: Vec): Float64Array {
  const v = new Float64Array(2 * masses.length)
  for (let i = 0; i < masses.length; i++) {
    v[2 * i] = p[2 * i] / masses[i]
    v[2 * i + 1] = p[2 * i + 1] / masses[i]
  }
  return v
}

/** The total energy H = T + V. */
export function energy(masses: readonly number[], q: Vec, p: Vec, softening = 0): number {
  return kineticEnergy(masses, p) + potentialEnergy(masses, q, softening)
}

/** The total linear momentum (P_x, P_y) = Σ_i p_i. */
export function momentum(masses: readonly number[], p: Vec): [number, number] {
  let px = 0
  let py = 0
  for (let i = 0; i < masses.length; i++) {
    px += p[2 * i]
    py += p[2 * i + 1]
  }
  return [px, py]
}

/** The angular momentum about the origin, L = Σ_i (x_i p_{y,i} − y_i p_{x,i}) (a scalar: the plane's z component). */
export function angularMomentum(masses: readonly number[], q: Vec, p: Vec): number {
  let l = 0
  for (let i = 0; i < masses.length; i++) l += q[2 * i] * p[2 * i + 1] - q[2 * i + 1] * p[2 * i]
  return l
}

/** The centre of mass (X, Y) = Σ_i m_i q_i / M. */
export function centreOfMass(masses: readonly number[], q: Vec): [number, number] {
  let m = 0
  let x = 0
  let y = 0
  for (let i = 0; i < masses.length; i++) {
    m += masses[i]
    x += masses[i] * q[2 * i]
    y += masses[i] * q[2 * i + 1]
  }
  return [x / m, y / m]
}

/**
 * The same state in the centre-of-mass frame: the centre of mass moved to the origin and its velocity P/M removed
 * from every body (p_i ← p_i − m_i P/M), so the total momentum is zero.
 */
export function toCentreOfMassFrame(masses: readonly number[], q: Vec, p: Vec): { q: Float64Array; p: Float64Array } {
  const [cx, cy] = centreOfMass(masses, q)
  const [px, py] = momentum(masses, p)
  const total = masses.reduce((s, m) => s + m, 0)
  const q2 = new Float64Array(2 * masses.length)
  const p2 = new Float64Array(2 * masses.length)
  for (let i = 0; i < masses.length; i++) {
    q2[2 * i] = q[2 * i] - cx
    q2[2 * i + 1] = q[2 * i + 1] - cy
    p2[2 * i] = p[2 * i] - (masses[i] * px) / total
    p2[2 * i + 1] = p[2 * i + 1] - (masses[i] * py) / total
  }
  return { q: q2, p: p2 }
}

/**
 * The N-body Hamiltonian as the engine's `SeparableHamiltonian`, with both gradients in closed form: for `symplectic`
 * (leapfrog, velocity Verlet) directly, or for any other solver through `hamiltonianSystem`, whose state is
 * x = (q, p) of length 4N.
 *
 * @param masses The masses m_i (N of them, positive).
 * @param softening The softening length ε (default 0, exact Newtonian gravity).
 */
export function nBodyHamiltonian(masses: readonly number[], softening = 0): SeparableHamiltonian {
  return {
    potential: (q) => potentialEnergy(masses, toFlat(q), softening),
    potentialGradient: (q) => potentialGradient(masses, toFlat(q), softening),
    kinetic: (p) => kineticEnergy(masses, toFlat(p)),
    kineticGradient: (p) => velocities(masses, toFlat(p)),
  }
}

/** The smallest pairwise distance min_{i<j} |q_i − q_j|, for spotting close encounters. */
export function closestPair(masses: readonly number[], q: Vec): number {
  let best = Infinity
  for (let i = 0; i < masses.length; i++)
    for (let j = i + 1; j < masses.length; j++)
      best = Math.min(best, Math.hypot(q[2 * i] - q[2 * j], q[2 * i + 1] - q[2 * j + 1]))
  return best
}

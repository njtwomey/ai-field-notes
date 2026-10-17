/**
 * The physics of the flattening toy: test particles of unit mass around a central mass in units where G = M = 1, with
 * a softened point-mass potential Φ(r) = −1/√(r² + ε²), and a pairwise drag between neighbours that stands in for
 * inelastic collisions. The drag on each pair acts along the line joining the two particles, as the impulse between two
 * smooth colliding spheres does: it removes the pair's approach or separation speed, conserves the pair's momentum, and
 * exerts no torque about any point, so the cloud's total angular momentum is conserved exactly by the forces.
 *
 * The engine does the numerics: `rungeKutta` integrates, `kdTree` and `treeSearch` find the neighbours, `eigh` gives
 * the second-moment tensor's eigenvalues.
 */
import { rungeKutta, type OdeState, type Rhs } from 'aifn-compute/dynamics/ode'
import { child, standardNormals, stream, units } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { eigh } from 'aifn-compute/numerics/linalg'
import { kdTree, treeSearch } from 'aifn-compute/numerics/neighbours'

/** The softening length ε of the central potential (length units, G = M = 1). */
export const SOFTENING = 0.2
/** The cloud's initial inner and outer radii. */
export const SHELL: readonly [number, number] = [0.5, 1.5]

/** Φ(r) = −1/√(r² + ε²). */
export const potential = (r2: number) => -1 / Math.sqrt(r2 + SOFTENING * SOFTENING)

/** The circular speed v_c(r) = √(r dΦ/dr) of the softened potential. */
export const circularSpeed = (r: number) => r / (r * r + SOFTENING * SOFTENING) ** 0.75

export type CloudOptions = {
  /** Number of particles n. */
  n: number
  /** Initial rotation as a fraction s of the local circular speed, about the z axis (0 to 1). */
  spin: number
  /** Drag rate γ per neighbour pair (1/time units). */
  dissipation: number
  /** Random velocity per component, as a fraction of the local circular speed. */
  dispersion: number
  /** Neighbours k searched per particle; each unordered pair found is used once. */
  neighbours: number
  /** RK4 step size h (time units). */
  stepSize: number
  /** Number of steps. */
  steps: number
  /** Steps between frames kept for the player. */
  every: number
  /** Steps between neighbour searches. */
  refresh: number
  seed: number
}

/** The state x = (positions, velocities): 3n positions then 3n velocities, particle-major. */
export function initialCloud({ n, spin, dispersion, seed }: Pick<CloudOptions, 'n' | 'spin' | 'dispersion' | 'seed'>) {
  const root = stream(seed)
  const dir = standardNormals(child(root, 'direction'), 3 * n)
  const rad = units(child(root, 'radius'), n)
  const kick = standardNormals(child(root, 'velocity'), 3 * n)
  const [a, b] = SHELL
  const x = new Float64Array(6 * n)
  for (let i = 0; i < n; i++) {
    // Uniform in volume within the shell: r³ uniform between a³ and b³.
    const r = Math.cbrt(a ** 3 + rad[i] * (b ** 3 - a ** 3))
    const d = Math.hypot(dir[3 * i], dir[3 * i + 1], dir[3 * i + 2]) || 1
    const px = (r * dir[3 * i]) / d
    const py = (r * dir[3 * i + 1]) / d
    const pz = (r * dir[3 * i + 2]) / d
    const vc = circularSpeed(r)
    x[3 * i] = px
    x[3 * i + 1] = py
    x[3 * i + 2] = pz
    // Rotation about z: s v_c(r) (ẑ × r)/r, so the speed is s v_c sin θ; then an isotropic random part.
    const o = 3 * n + 3 * i
    x[o] = (spin * vc * -py) / r + dispersion * vc * kick[3 * i]
    x[o + 1] = (spin * vc * px) / r + dispersion * vc * kick[3 * i + 1]
    x[o + 2] = dispersion * vc * kick[3 * i + 2]
    // Keep every particle bound: a speed above 0.9 of the escape speed √(−2Φ) is scaled down to it.
    const speed = Math.hypot(x[o], x[o + 1], x[o + 2])
    const cap = 0.9 * Math.sqrt(-2 * potential(r * r))
    if (speed > cap) for (let c = 0; c < 3; c++) x[o + c] *= cap / speed
  }
  return x
}

/** The unordered neighbour pairs (i < j) among each particle's k nearest, flattened as [i₀, j₀, i₁, j₁, …]. */
export function neighbourPairs(x: ArrayLike<number>, n: number, k: number): Int32Array {
  const rows = Array.from({ length: n }, (_, i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]])
  const found = toFlat(treeSearch(kdTree(rows, { leafSize: 16 }), rows, k + 1).indices)
  const seen = new Set<number>()
  const out: number[] = []
  for (let i = 0; i < n; i++)
    for (let m = 0; m < k + 1; m++) {
      const j = found[i * (k + 1) + m]
      if (j === i) continue
      const lo = Math.min(i, j)
      const hi = Math.max(i, j)
      const key = lo * n + hi
      if (seen.has(key)) continue
      seen.add(key)
      out.push(lo, hi)
    }
  return Int32Array.from(out)
}

/**
 * The drag accelerations a_i = −γ Σ_j [(v_i − v_j)·n̂_ij] n̂_ij over the pairs, with n̂_ij the unit vector from j to i.
 * Each pair adds equal and opposite forces along n̂_ij, so the momentum and the torque about the origin of each pair's
 * forces cancel. Written into `out` (length 3n), which is added to.
 */
export function addDrag(x: ArrayLike<number>, n: number, pairs: Int32Array, gamma: number, out: Float64Array) {
  if (gamma === 0) return
  const v = 3 * n
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p]
    const j = pairs[p + 1]
    const dx = x[3 * i] - x[3 * j]
    const dy = x[3 * i + 1] - x[3 * j + 1]
    const dz = x[3 * i + 2] - x[3 * j + 2]
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 === 0) continue
    const dvx = x[v + 3 * i] - x[v + 3 * j]
    const dvy = x[v + 3 * i + 1] - x[v + 3 * j + 1]
    const dvz = x[v + 3 * i + 2] - x[v + 3 * j + 2]
    // −γ (Δv·Δr / |Δr|²) Δr: the projection of the relative velocity on the line of centres.
    const c = (-gamma * (dvx * dx + dvy * dy + dvz * dz)) / d2
    out[3 * i] += c * dx
    out[3 * i + 1] += c * dy
    out[3 * i + 2] += c * dz
    out[3 * j] -= c * dx
    out[3 * j + 1] -= c * dy
    out[3 * j + 2] -= c * dz
  }
}

/** The right-hand side f(t, x): positions move with the velocities; velocities change by gravity plus drag. */
export function cloudRhs(n: number, gamma: number, pairs: { current: Int32Array }): Rhs {
  const eps2 = SOFTENING * SOFTENING
  return (_t, state) => {
    const x = toFlat(state)
    const f = new Float64Array(6 * n)
    const acc = new Float64Array(3 * n)
    for (let i = 0; i < n; i++) {
      const px = x[3 * i]
      const py = x[3 * i + 1]
      const pz = x[3 * i + 2]
      const s = (px * px + py * py + pz * pz + eps2) ** -1.5
      acc[3 * i] = -px * s
      acc[3 * i + 1] = -py * s
      acc[3 * i + 2] = -pz * s
    }
    addDrag(x, n, pairs.current, gamma, acc)
    for (let k = 0; k < 3 * n; k++) {
      f[k] = x[3 * n + k]
      f[3 * n + k] = acc[k]
    }
    return f
  }
}

export type Diagnostics = {
  /** Energy per particle, Σ(½|v|² + Φ)/n. */
  energy: number
  /** Total angular momentum per particle, Σ r × v / n (a 3-vector). */
  angularMomentum: [number, number, number]
  /** The smallest over the largest axis of the cloud, √(λ_min/λ_max) of the second-moment tensor Σ r rᵀ / n. */
  axisRatio: number
  /** The root-mean-square cylindrical radius √⟨x² + y²⟩. */
  radius: number
  /** The rms of the height z. */
  height: number
  /** The magnitude of the net torque Σ r_i × F_i of the drag forces, per particle. */
  dragTorque: number
}

export function diagnostics(x: ArrayLike<number>, n: number, pairs: Int32Array, gamma: number): Diagnostics {
  const v = 3 * n
  let e = 0
  let lx = 0
  let ly = 0
  let lz = 0
  const m = [0, 0, 0, 0, 0, 0] // xx, xy, xz, yy, yz, zz
  for (let i = 0; i < n; i++) {
    const [px, py, pz] = [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
    const [ux, uy, uz] = [x[v + 3 * i], x[v + 3 * i + 1], x[v + 3 * i + 2]]
    e += 0.5 * (ux * ux + uy * uy + uz * uz) + potential(px * px + py * py + pz * pz)
    lx += py * uz - pz * uy
    ly += pz * ux - px * uz
    lz += px * uy - py * ux
    m[0] += px * px
    m[1] += px * py
    m[2] += px * pz
    m[3] += py * py
    m[4] += py * pz
    m[5] += pz * pz
  }
  const M = [
    [m[0] / n, m[1] / n, m[2] / n],
    [m[1] / n, m[3] / n, m[4] / n],
    [m[2] / n, m[4] / n, m[5] / n],
  ]
  const lambda = toFlat(eigh(tensor(M)).values)
  const drag = new Float64Array(3 * n)
  addDrag(x, n, pairs, gamma, drag)
  let tx = 0
  let ty = 0
  let tz = 0
  for (let i = 0; i < n; i++) {
    const [px, py, pz] = [x[3 * i], x[3 * i + 1], x[3 * i + 2]]
    const [fx, fy, fz] = [drag[3 * i], drag[3 * i + 1], drag[3 * i + 2]]
    tx += py * fz - pz * fy
    ty += pz * fx - px * fz
    tz += px * fy - py * fx
  }
  return {
    energy: e / n,
    angularMomentum: [lx / n, ly / n, lz / n],
    axisRatio: Math.sqrt(Math.max(0, lambda[2]) / lambda[0]),
    radius: Math.sqrt(M[0][0] + M[1][1]),
    height: Math.sqrt(M[2][2]),
    dragTorque: Math.hypot(tx, ty, tz) / n,
  }
}

export type CloudFrame = Diagnostics & {
  time: number
  /** Positions, 3n, particle-major. */
  positions: Float64Array
}

/**
 * The whole run, computed up front: RK4 steps of size h, the neighbour pairs found again every `refresh` steps (and
 * held fixed within a step, so every stage's drag is torque-free), and a frame kept every `every` steps.
 */
export function simulateCloud(o: CloudOptions): CloudFrame[] {
  const { n, dissipation: gamma, neighbours: k } = o
  const x0 = initialCloud(o)
  const pairs = { current: neighbourPairs(x0, n, k) }
  const solver = rungeKutta(cloudRhs(n, gamma, pairs), 'rk4', { stepSize: o.stepSize })
  const s = stream(o.seed)
  let state: OdeState = solver.init({ x0 }, s)
  const frame = (st: OdeState): CloudFrame => {
    const x = toFlat(st.x)
    return {
      time: st.time,
      positions: Float64Array.from(x.slice(0, 3 * n)),
      ...diagnostics(x, n, pairs.current, gamma),
    }
  }
  const frames = [frame(state)]
  for (let step = 0; step < o.steps; step++) {
    if (step > 0 && step % o.refresh === 0) pairs.current = neighbourPairs(toFlat(state.x), n, k)
    state = solver.step(state, { t: step, stream: s })
    if (state.diverged) break
    if ((step + 1) % o.every === 0) frames.push(frame(state))
  }
  return frames
}

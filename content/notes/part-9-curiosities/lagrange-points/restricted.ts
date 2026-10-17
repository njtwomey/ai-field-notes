/**
 * The circular restricted three-body problem in the frame that rotates with the two heavy bodies.
 *
 * Units: G = 1, total mass m₁ + m₂ = 1, separation 1, so the angular velocity of the frame is n = 1 and one orbit of
 * the primaries takes 2π. The mass ratio is μ = m₂ / (m₁ + m₂) ∈ (0, 1/2]; m₁ = 1 − μ sits at (−μ, 0) and m₂ = μ at
 * (1 − μ, 0). A massless test particle at (x, y) with rotating-frame velocity (ẋ, ẏ) obeys
 *   ẍ − 2ẏ = ∂Ω/∂x,   ÿ + 2ẋ = ∂Ω/∂y,
 *   Ω(x, y) = (x² + y²)/2 + (1 − μ)/r₁ + μ/r₂,
 * with r₁ and r₂ its distances to m₁ and m₂ (Murray and Dermott, 1999, chapter 3). The Jacobi
 * constant C = 2Ω − (ẋ² + ẏ²) is conserved.
 *
 * Plain physics, no React. Root-finding is the engine's (`findRoot`, Brent's method).
 */
import { findRoot } from 'aifn-compute/numerics/roots'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { dormandPrince, type Rhs } from 'aifn-compute/dynamics/ode'

/** Ω(x, y), the effective potential of the rotating frame (positive; 2Ω is the zero-velocity value of C). */
export function effectivePotential(x: number, y: number, mu: number): number {
  const r1 = Math.hypot(x + mu, y)
  const r2 = Math.hypot(x - 1 + mu, y)
  return (x * x + y * y) / 2 + (1 - mu) / r1 + mu / r2
}

/** (∂Ω/∂x, ∂Ω/∂y). */
export function effectiveGradient(x: number, y: number, mu: number): [number, number] {
  const dx1 = x + mu
  const dx2 = x - 1 + mu
  const r1 = Math.hypot(dx1, y)
  const r2 = Math.hypot(dx2, y)
  const a = (1 - mu) / (r1 * r1 * r1)
  const b = mu / (r2 * r2 * r2)
  return [x - a * dx1 - b * dx2, y - a * y - b * y]
}

/** The Jacobi constant C = 2Ω(x, y) − (ẋ² + ẏ²) of a state (x, y, ẋ, ẏ). */
export function jacobiConstant(x: number, y: number, vx: number, vy: number, mu: number): number {
  return 2 * effectivePotential(x, y, mu) - vx * vx - vy * vy
}

/** The right-hand side for a state (x, y, ẋ, ẏ): the Coriolis terms ±2ẏ, ∓2ẋ plus the gradient of Ω. */
export function restrictedRhs(mu: number): Rhs {
  return (_t, s) => {
    const [x, y, vx, vy] = toFlat(s)
    const [gx, gy] = effectiveGradient(x, y, mu)
    return [vx, vy, 2 * vy + gx, -2 * vx + gy]
  }
}

/** ∂Ω/∂x on the x axis (y = 0), whose three zeros are L1, L2 and L3. */
export function axisForce(x: number, mu: number): number {
  return effectiveGradient(x, 0, mu)[0]
}

export type LagrangePoint = { name: string; x: number; y: number; C: number }

/**
 * The five equilibria. L1, L2 and L3 are the zeros of ∂Ω/∂x on the axis, one in each interval the masses cut it
 * into: L1 between them, L2 beyond m₂, L3 beyond m₁, each bracketed and found by Brent's method. L4 and L5 are the
 * apexes of the equilateral triangles on the segment m₁m₂: (1/2 − μ, ±√3/2).
 */
export function lagrangePoints(mu: number): LagrangePoint[] {
  const eps = 1e-9
  const f = (x: number) => axisForce(x, mu)
  const brackets: [string, [number, number]][] = [
    ['L1', [-mu + eps, 1 - mu - eps]],
    ['L2', [1 - mu + eps, 2]],
    ['L3', [-2, -mu - eps]],
  ]
  const collinear = brackets.map(([name, b]) => {
    const x = findRoot(f, b, { xtol: 1e-14 }).x
    return { name, x, y: 0, C: 2 * effectivePotential(x, 0, mu) }
  })
  const xt = 0.5 - mu
  const yt = Math.sqrt(3) / 2
  const ct = 2 * effectivePotential(xt, yt, mu)
  return [...collinear, { name: 'L4', x: xt, y: yt, C: ct }, { name: 'L5', x: xt, y: -yt, C: ct }]
}

/** Routh's critical mass ratio: L4 and L5 are linearly stable when 27μ(1 − μ) < 1, i.e. μ < (1 − √(23/27))/2. */
export const ROUTH_MU = (1 - Math.sqrt(23 / 27)) / 2

/** n evenly spaced values from lo to hi. */
export const linspace = (lo: number, hi: number, n: number) =>
  Array.from({ length: n }, (_, k) => lo + ((hi - lo) * k) / (n - 1))

/**
 * 2Ω on a grid (row-major: z[i][j] at (xs[j], ys[i])), capped at `cap` so the poles at the two masses do not swamp
 * the colour scale. A level set 2Ω = C of this grid is the zero-velocity curve of Jacobi constant C.
 */
export function potentialGrid(mu: number, xs: readonly number[], ys: readonly number[], cap: number): number[][] {
  return ys.map((y) => xs.map((x) => Math.min(cap, 2 * effectivePotential(x, y, mu))))
}

export type ParticleRun = {
  time: number[]
  x: number[]
  y: number[]
  /** The Jacobi constant at the start. */
  jacobi: number
  /** The largest |C(t) − C(0)| over the run. */
  jacobiDrift: number
  /** Why the run stopped early, or null. */
  failure: string | null
}

/**
 * A test particle from (x0, y0) with rotating-frame velocity (vx0, vy0), integrated to `tEnd` by Dormand–Prince with
 * relative tolerance `rtol` (absolute rtol/100). Stops early on a step failure, after `maxSteps` steps, or when the
 * particle comes within 10⁻³ of a mass (a collision for this toy).
 */
export function runParticle(
  mu: number,
  [x0, y0, vx0, vy0]: readonly [number, number, number, number],
  tEnd: number,
  rtol: number,
  maxSteps = 100_000,
): ParticleRun {
  const jacobi = jacobiConstant(x0, y0, vx0, vy0, mu)
  const run: ParticleRun = { time: [], x: [], y: [], jacobi, jacobiDrift: 0, failure: null }
  const solver = dormandPrince(restrictedRhs(mu), { tEnd, rtol, atol: rtol / 100 })
  let steps = 0
  for (const { state } of live(solver, { x0: [x0, y0, vx0, vy0] })) {
    const [x, y, vx, vy] = toFlat(state.x)
    run.time.push(state.time)
    run.x.push(x)
    run.y.push(y)
    run.jacobiDrift = Math.max(run.jacobiDrift, Math.abs(jacobiConstant(x, y, vx, vy, mu) - jacobi))
    if (state.diverged) {
      run.failure = state.failure
      break
    }
    if (Math.hypot(x + mu, y) < 1e-3 || Math.hypot(x - 1 + mu, y) < 1e-3) {
      run.failure = 'hit a mass'
      break
    }
    if (++steps >= maxSteps) {
      run.failure = `stopped after ${maxSteps} steps`
      break
    }
  }
  return run
}

/**
 * Hamiltonian Monte Carlo (Duane et al., 1987; Neal, 2011) with the leapfrog integrator, and the No-U-Turn Sampler
 * (Hoffman & Gelman, 2014, Algorithm 3). Both expose the trajectory, the momenta and the energy along it.
 *
 * Conventions: the potential is U(θ) = −log π(θ), the kinetic energy K(p) = ½ Σ pᵢ²/mᵢ for a diagonal mass matrix
 * M = diag(m), momenta are drawn p ~ N(0, M), and the Hamiltonian H = U + K is the energy.
 */

import { child, uniform } from 'aifn/foundation/random'
import type { Matrix, Vector } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import { badLogDensity } from './metropolis'
import type { AcceptRejectState, ChainStart, LogDensity } from './types'
import { allFinite, data, logDensityAndGrad, mat, perCoordinate, standardNormals, toF64, vec, type F64 } from './util'

/** A leapfrog trajectory: positions and momenta at each of its L + 1 points, and H at each. */
export type Leapfrog = {
  positions: Matrix
  momenta: Matrix
  /** H(θ, p) at each point. */
  energies: Vector
  /** log π at each point. */
  logDensities: Vector
  /** The gradient ∇ log π at the last point. */
  finalGrad: Vector
  /** False when a position, momentum or energy became non-finite (the trajectory stops there). */
  finite: boolean
}

/** Kinetic energy ½ Σ pᵢ²/mᵢ. */
function kinetic(p: F64, mass: F64): number {
  let k = 0
  for (let i = 0; i < p.length; i++) k += (p[i] * p[i]) / mass[i]
  return 0.5 * k
}

/** One leapfrog step of size ε from (θ, p) with gradient g = ∇ log π(θ): half momentum, full position, half momentum. */
function leapfrogStep(target: LogDensity, theta: F64, p: F64, g: F64, eps: number, mass: F64) {
  const d = theta.length
  const pHalf = new Float64Array(d)
  const next = new Float64Array(d)
  for (let i = 0; i < d; i++) {
    pHalf[i] = p[i] + 0.5 * eps * g[i]
    next[i] = theta[i] + (eps * pHalf[i]) / mass[i]
  }
  const { value, grad } = logDensityAndGrad(target, next)
  const pNext = new Float64Array(d)
  for (let i = 0; i < d; i++) pNext[i] = pHalf[i] + 0.5 * eps * grad[i]
  return { theta: next, p: pNext, logDensity: value, grad }
}

/**
 * Integrate Hamilton's equations with `steps` leapfrog steps of size `stepSize` from (θ₀, p₀) (Neal, 2011, eq. 5.18).
 * The leapfrog map is volume-preserving and reversible, and its energy error stays bounded for a stable step size
 * (ε below about 2/√(largest precision eigenvalue) for a Gaussian), which is why HMC accepts most proposals.
 */
export function leapfrog(
  target: LogDensity,
  theta0: Vector | ArrayLike<number>,
  momentum0: Vector | ArrayLike<number>,
  options: { stepSize: number; steps: number; mass?: number | ArrayLike<number> },
): Leapfrog {
  const d = target.dim
  const mass = perCoordinate(options.mass ?? 1, d, 'leapfrog')
  let theta = toF64(theta0, 'leapfrog')
  let p = toF64(momentum0, 'leapfrog')
  let { value, grad } = logDensityAndGrad(target, theta)
  const positions: F64[] = [theta]
  const momenta: F64[] = [p]
  const energies: number[] = [-value + kinetic(p, mass)]
  const logs: number[] = [value]
  let finite = Number.isFinite(energies[0])
  for (let l = 0; l < options.steps && finite; l++) {
    const next = leapfrogStep(target, theta, p, grad, options.stepSize, mass)
    ;({ theta, p, grad } = next)
    value = next.logDensity
    const h = -value + kinetic(p, mass)
    positions.push(theta)
    momenta.push(p)
    energies.push(h)
    logs.push(value)
    finite = Number.isFinite(h) && allFinite(theta)
  }
  const flat = (rows: F64[]) => {
    const out = new Float64Array(rows.length * d)
    rows.forEach((r, i) => out.set(r, i * d))
    return mat(out, rows.length, d)
  }
  return {
    positions: flat(positions),
    momenta: flat(momenta),
    energies: vec(Float64Array.from(energies)),
    logDensities: vec(Float64Array.from(logs)),
    finalGrad: vec(grad),
    finite,
  }
}

/** The state of `hmc`. `proposal` is the trajectory's end point. */
export type HmcState = AcceptRejectState & {
  /** ∇ log π(x). */
  grad: Vector
  /** The momentum drawn on the last step (zeros at t = 0). */
  momentum: Vector
  /** The last trajectory, (L + 1)×d, starting at the previous x; its momenta; H along it. */
  trajectory: Matrix
  trajectoryMomenta: Matrix
  energies: Vector
  /** H(end) − H(start) of the last trajectory; the log acceptance ratio is its negative. */
  energyError: number
  /** True when the last trajectory's energy error exceeded `divergenceThreshold` or became non-finite. */
  divergent: boolean
  divergentCount: number
  /** Gradient evaluations so far. */
  gradientEvaluations: number
  /** The step size used on the last step (after jitter). */
  stepSize: number
}

/** Options for `hmc`. */
export type HmcOptions = {
  /** Leapfrog step size ε. Default 0.1. */
  stepSize?: number
  /** Leapfrog steps L per proposal. Default 20. */
  steps?: number
  /** Diagonal of the mass matrix M (one number or one per coordinate). Default 1. */
  mass?: number | ArrayLike<number>
  /** Draw ε uniformly in [ε(1 − j), ε(1 + j)] each step, to avoid periodic trajectories (Neal, 2011, §5.4.2). Default 0. */
  stepJitter?: number
  /** Flag a trajectory as divergent when its energy error exceeds this (Stan's 1000). Default 1000. */
  divergenceThreshold?: number
}

function hmcStart(target: LogDensity, x0: F64, name: string) {
  if (x0.length !== target.dim) throw new Error(`${name}: x0 has ${x0.length} values for dimension ${target.dim}`)
  const { value, grad } = logDensityAndGrad(target, x0)
  return { value, grad, diverged: badLogDensity(value) || !allFinite(x0) }
}

/**
 * Hamiltonian Monte Carlo (Duane et al., 1987; Neal, 2011, §5.3.2): draw p ~ N(0, M), run L leapfrog steps of size ε
 * from (x, p), and accept the end point with probability min(1, exp(−ΔH)). Step t draws the momentum from
 * `child(ctx.stream, 'momentum')` and the uniform from the step stream `ctx.stream` itself.
 */
export function hmc(target: LogDensity, options: HmcOptions = {}): Algorithm<ChainStart, HmcState> {
  const name = 'hmc'
  const { stepSize = 0.1, steps = 20, stepJitter = 0, divergenceThreshold = 1000 } = options
  const d = target.dim
  const mass = perCoordinate(options.mass ?? 1, d, name)
  const sqrtMass = mass.map(Math.sqrt)
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const start = hmcStart(target, x, name)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: start.value,
        grad: vec(start.grad),
        proposal: X,
        proposalLogDensity: start.value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        momentum: vec(new Float64Array(d)),
        trajectory: mat(Float64Array.from(x), 1, d),
        trajectoryMomenta: mat(new Float64Array(d), 1, d),
        energies: vec(Float64Array.of(-start.value)),
        energyError: 0,
        divergent: false,
        divergentCount: 0,
        gradientEvaluations: 1,
        stepSize,
        diverged: start.diverged,
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const z = standardNormals(child(draws, 'momentum'), d)
      const p = z.map((zi, i) => zi * sqrtMass[i])
      const eps = stepJitter > 0 ? stepSize * (1 + stepJitter * (2 * uniform(child(draws, 'jitter')) - 1)) : stepSize
      const path = leapfrog(target, s.x, p, { stepSize: eps, steps, mass })
      const H = data(path.energies)
      const last = H.length - 1
      const energyError = path.finite ? H[last] - H[0] : Infinity
      const divergent = !path.finite || Math.abs(energyError) > divergenceThreshold
      const logRatio = path.finite ? -energyError : -Infinity
      const acceptance = Math.min(1, Math.exp(logRatio))
      const accepted = uniform(draws) < acceptance
      const P = data(path.positions)
      const end = P.slice(last * d, (last + 1) * d)
      const endLog = data(path.logDensities)[last]
      const acceptedCount = s.acceptedCount + (accepted ? 1 : 0)
      return {
        ...s,
        t: s.t + 1,
        x: accepted ? vec(end) : s.x,
        logDensity: accepted ? endLog : s.logDensity,
        grad: accepted ? path.finalGrad : s.grad,
        proposal: vec(end),
        proposalLogDensity: endLog,
        logAcceptanceRatio: logRatio,
        acceptance,
        accepted,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        momentum: vec(p),
        trajectory: path.positions,
        trajectoryMomenta: path.momenta,
        energies: path.energies,
        energyError,
        divergent,
        divergentCount: s.divergentCount + (divergent ? 1 : 0),
        gradientEvaluations: s.gradientEvaluations + last,
        stepSize: eps,
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// NUTS.

/** The state of `nuts`. */
export type NutsState = AcceptRejectState & {
  grad: Vector
  momentum: Vector
  /** Every point the last tree visited, ordered by integration time (leftmost first), and their times (in steps). */
  trajectory: Matrix
  trajectoryTimes: Vector
  /** H at each trajectory point. */
  energies: Vector
  /** Depth of the last tree (2^depth leapfrog steps, fewer when it stopped early) and its leapfrog steps. */
  treeDepth: number
  leapfrogSteps: number
  /** True when the last tree hit an energy error above `divergenceThreshold` (Hoffman & Gelman's Δmax). */
  divergent: boolean
  divergentCount: number
  gradientEvaluations: number
  /** True when the last tree stopped at `maxDepth` rather than by a U-turn. */
  hitMaxDepth: boolean
}

/** Options for `nuts`. */
export type NutsOptions = {
  /** Leapfrog step size ε. Default 0.1. */
  stepSize?: number
  /** Largest tree depth (at most 2^maxDepth leapfrog steps). Default 10. */
  maxDepth?: number
  mass?: number | ArrayLike<number>
  /** Δmax: stop building when the energy error passes this. Default 1000. */
  divergenceThreshold?: number
}

type Point = { theta: F64; p: F64; grad: F64; logDensity: number }
type Tree = {
  minus: Point
  plus: Point
  candidate: Point
  n: number
  keepGoing: boolean
  divergent: boolean
}

/**
 * The No-U-Turn Sampler (Hoffman & Gelman, 2014, Algorithm 3, "efficient NUTS"): a slice variable u ~ U(0, e^{−H})
 * picks the acceptable points; the trajectory doubles forwards or backwards at random until its ends start to turn
 * back ((θ⁺ − θ⁻)·M⁻¹p < 0 at either end), and the next point is drawn uniformly from the acceptable points, favouring
 * the newest subtree. The step size is fixed (no dual averaging). All draws of step t come from its step stream `ctx.stream`.
 */
export function nuts(target: LogDensity, options: NutsOptions = {}): Algorithm<ChainStart, NutsState> {
  const name = 'nuts'
  const { stepSize = 0.1, maxDepth = 10, divergenceThreshold = 1000 } = options
  const d = target.dim
  const mass = perCoordinate(options.mass ?? 1, d, name)
  const sqrtMass = mass.map(Math.sqrt)

  const noUTurn = (minus: F64, plus: F64, p: F64) => {
    let s = 0
    for (let i = 0; i < d; i++) s += ((plus[i] - minus[i]) * p[i]) / mass[i]
    return s >= 0
  }

  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const start = hmcStart(target, x, name)
      const X = vec(x)
      return {
        t: 0,
        x: X,
        logDensity: start.value,
        grad: vec(start.grad),
        proposal: X,
        proposalLogDensity: start.value,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: false,
        acceptedCount: 0,
        acceptanceRate: NaN,
        momentum: vec(new Float64Array(d)),
        trajectory: mat(Float64Array.from(x), 1, d),
        trajectoryTimes: vec(Float64Array.of(0)),
        energies: vec(Float64Array.of(-start.value)),
        treeDepth: 0,
        leapfrogSteps: 0,
        divergent: false,
        divergentCount: 0,
        gradientEvaluations: 1,
        hitMaxDepth: false,
        diverged: start.diverged,
      }
    },
    step: (s, ctx) => {
      const draws = ctx.stream
      const z = standardNormals(child(draws, 'momentum'), d)
      const p0 = z.map((zi, i) => zi * sqrtMass[i])
      const x0 = data(s.x)
      const joint0 = s.logDensity - kinetic(p0, mass)
      // log u with u ~ U(0, exp(joint0)).
      const logU = joint0 + Math.log(uniform(draws))
      const visited: { time: number; theta: F64; energy: number }[] = [{ time: 0, theta: x0, energy: -joint0 }]
      let steps = 0

      const buildTree = (from: Point, direction: 1 | -1, depth: number, time: number): Tree & { time: number } => {
        if (depth === 0) {
          const next = leapfrogStep(target, from.theta, from.p, from.grad, direction * stepSize, mass)
          steps++
          const joint = next.logDensity - kinetic(next.p, mass)
          const nextTime = time + direction
          visited.push({ time: nextTime, theta: next.theta, energy: -joint })
          const point: Point = { theta: next.theta, p: next.p, grad: next.grad, logDensity: next.logDensity }
          const ok = Number.isFinite(joint)
          const divergent = !ok || logU - joint >= divergenceThreshold
          return {
            minus: point,
            plus: point,
            candidate: point,
            n: ok && logU <= joint ? 1 : 0,
            keepGoing: !divergent,
            divergent,
            time: nextTime,
          }
        }
        const first = buildTree(from, direction, depth - 1, time)
        if (!first.keepGoing) return first
        const edge = direction === -1 ? first.minus : first.plus
        const second = buildTree(edge, direction, depth - 1, first.time)
        const n = first.n + second.n
        const candidate = n > 0 && uniform(draws) < second.n / n ? second.candidate : first.candidate
        const minus = direction === -1 ? second.minus : first.minus
        const plus = direction === -1 ? first.plus : second.plus
        return {
          minus,
          plus,
          candidate,
          n,
          keepGoing:
            second.keepGoing && noUTurn(minus.theta, plus.theta, minus.p) && noUTurn(minus.theta, plus.theta, plus.p),
          divergent: second.divergent,
          time: second.time,
        }
      }

      const here: Point = { theta: x0, p: p0, grad: data(s.grad), logDensity: s.logDensity }
      let minus = here
      let plus = here
      let timeMinus = 0
      let timePlus = 0
      let chosen = here
      let n = 1
      let depth = 0
      let keepGoing = true
      let divergent = false
      while (keepGoing && depth < maxDepth) {
        const direction: 1 | -1 = uniform(draws) < 0.5 ? -1 : 1
        const tree = buildTree(
          direction === -1 ? minus : plus,
          direction,
          depth,
          direction === -1 ? timeMinus : timePlus,
        )
        if (direction === -1) {
          minus = tree.minus
          timeMinus = tree.time
        } else {
          plus = tree.plus
          timePlus = tree.time
        }
        divergent ||= tree.divergent
        if (tree.keepGoing && uniform(draws) < Math.min(1, tree.n / n)) chosen = tree.candidate
        n += tree.n
        keepGoing =
          tree.keepGoing && noUTurn(minus.theta, plus.theta, minus.p) && noUTurn(minus.theta, plus.theta, plus.p)
        depth++
      }
      visited.sort((a, b) => a.time - b.time)
      const trajectory = new Float64Array(visited.length * d)
      visited.forEach((v, i) => trajectory.set(v.theta, i * d))
      const moved = chosen !== here
      const acceptedCount = s.acceptedCount + (moved ? 1 : 0)
      return {
        ...s,
        t: s.t + 1,
        x: vec(chosen.theta),
        logDensity: chosen.logDensity,
        grad: vec(chosen.grad),
        proposal: vec(chosen.theta),
        proposalLogDensity: chosen.logDensity,
        logAcceptanceRatio: NaN,
        acceptance: NaN,
        accepted: moved,
        acceptedCount,
        acceptanceRate: acceptedCount / (s.t + 1),
        momentum: vec(p0),
        trajectory: mat(trajectory, visited.length, d),
        trajectoryTimes: vec(Float64Array.from(visited, (v) => v.time)),
        energies: vec(Float64Array.from(visited, (v) => v.energy)),
        treeDepth: depth,
        leapfrogSteps: steps,
        divergent,
        divergentCount: s.divergentCount + (divergent ? 1 : 0),
        gradientEvaluations: s.gradientEvaluations + steps,
        hitMaxDepth: keepGoing && depth >= maxDepth,
      }
    },
  }
}

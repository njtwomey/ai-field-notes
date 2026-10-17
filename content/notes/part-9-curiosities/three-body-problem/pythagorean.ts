/**
 * Burrau's Pythagorean three-body problem (Szebehely and Peters, 1967): masses 3, 4 and 5 at rest at the vertices of
 * a 3-4-5 right triangle, each mass opposite the side of its own length, G = 1. Two copies are integrated together as
 * one system, the second with mass 3 moved by δ along x, so that both share every accepted step of the adaptive
 * solver and their separation is read at the same times.
 *
 * Plain physics and bookkeeping, no React. The integration is the engine's Dormand–Prince 5(4) pair.
 */
import { dormandPrince, type Rhs } from 'aifn-compute/dynamics/ode'
import { live } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { closestPair, energy, potentialGradient, velocities } from '../_shared/gravity'

export const MASSES = [3, 4, 5] as const
/** Mass 3 at (1, 3), mass 4 at (−2, −1), mass 5 at (1, −1); the centre of mass is at the origin. */
export const START = [1, 3, -2, -1, 1, -1] as const

const N = MASSES.length
const D = 2 * N // positions (or momenta) of one copy

/** The derivative of two independent copies, each state (q, p): q′ = p/m, p′ = −∇V(q). Length 4D. */
export function twinRhs(softening: number): Rhs {
  const m = [...MASSES]
  return (_t, x) => {
    const a = toFlat(x)
    const out = new Float64Array(4 * D)
    for (const o of [0, 2 * D]) {
      const q = a.slice(o, o + D)
      const p = a.slice(o + D, o + 2 * D)
      out.set(velocities(m, p), o)
      const g = potentialGradient(m, q, softening)
      for (let k = 0; k < D; k++) out[o + D + k] = -g[k]
    }
    return out
  }
}

export type PythagoreanRun = {
  /** The time of every accepted step, from 0. */
  time: number[]
  /** Positions of copy A at every step (D values each). */
  qa: number[][]
  /** Positions of copy B. */
  qb: number[][]
  /** |q_A − q_B|, the Euclidean norm over all six coordinates. */
  separation: number[]
  /** |E_A(t)/E_A(0) − 1|, the relative energy error of copy A. */
  energyError: number[]
  /** The smallest pairwise distance in copy A at each step. */
  closest: number[]
  /** Why the run ended early, or null. */
  failure: string | null
  /** The energy of copy A at t = 0. */
  e0: number
}

/**
 * Integrate both copies from rest to `tEnd` with relative tolerance `rtol` (absolute tolerance rtol/100), stopping
 * after `maxSteps` accepted steps.
 */
export function runPythagorean({
  tEnd,
  rtol,
  delta,
  softening,
  start = START,
  maxSteps = 250_000,
}: {
  tEnd: number
  rtol: number
  delta: number
  softening: number
  /** The starting positions (D values); the bodies start at rest. Default Burrau's triangle. */
  start?: readonly number[]
  maxSteps?: number
}): PythagoreanRun {
  const m = [...MASSES]
  const zeros = new Array(D).fill(0)
  const qa0 = [...start]
  const qb0 = [start[0] + delta, ...start.slice(1)]
  const e0 = energy(m, qa0, zeros, softening)
  const run: PythagoreanRun = {
    time: [],
    qa: [],
    qb: [],
    separation: [],
    energyError: [],
    closest: [],
    failure: null,
    e0,
  }
  const solver = dormandPrince(twinRhs(softening), { tEnd, rtol, atol: rtol / 100 })
  let steps = 0
  for (const { state } of live(solver, { x0: [...qa0, ...zeros, ...qb0, ...zeros] })) {
    const a = toFlat(state.x)
    const qa = a.slice(0, D)
    const qb = a.slice(2 * D, 3 * D)
    let s = 0
    for (let k = 0; k < D; k++) s += (qa[k] - qb[k]) ** 2
    run.time.push(state.time)
    run.qa.push(qa)
    run.qb.push(qb)
    run.separation.push(Math.sqrt(s))
    run.energyError.push(Math.abs(energy(m, qa, a.slice(D, 2 * D), softening) / e0 - 1))
    run.closest.push(closestPair(m, qa))
    if (state.diverged) {
      run.failure = state.failure
      break
    }
    if (++steps >= maxSteps) {
      run.failure = `stopped after ${maxSteps} steps`
      break
    }
  }
  return run
}

/** For each frame time k·dt, the last accepted step at or before it. */
export function frameSteps(time: readonly number[], dt: number): number[] {
  const last = time[time.length - 1]
  const frames = Math.floor(last / dt) + 1
  const out: number[] = []
  let i = 0
  for (let k = 0; k < frames; k++) {
    while (i + 1 < time.length && time[i + 1] <= k * dt) i++
    out.push(i)
  }
  return out
}

/** The steps to draw a path through: a step is kept once some body has moved `minMove` since the last kept one. */
export function pathSteps(q: readonly number[][], minMove: number): number[] {
  const kept = [0]
  let ref = q[0]
  for (let i = 1; i < q.length; i++) {
    let far = i === q.length - 1
    for (let b = 0; b < N && !far; b++)
      far = Math.hypot(q[i][2 * b] - ref[2 * b], q[i][2 * b + 1] - ref[2 * b + 1]) > minMove
    if (far) {
      kept.push(i)
      ref = q[i]
    }
  }
  return kept
}

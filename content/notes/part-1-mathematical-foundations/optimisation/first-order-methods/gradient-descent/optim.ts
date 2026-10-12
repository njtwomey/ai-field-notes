/** Gradient descent and heavy-ball momentum runs for the widgets in this note, on the engine's optimisers. */
import { dense } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { gradientDescent, momentum, type ObjectiveFn } from 'aifn-compute/optim'
import type { Vec } from './surfaces'

export type Trajectory = { path: Vec[]; losses: number[]; diverged: boolean }

/** Loss above which a run stops and is reported as diverged. */
export const DIVERGED = 1e8

/**
 * `steps` steps from `start` with step size `lr`: plain gradient descent when `beta` is 0, else heavy-ball momentum
 * (the engine's v ← βv + ∇L, θ ← θ − ηv, which is the note's v ← βv − η∇L, θ ← θ + v with v rescaled by −η).
 * The gradient-norm test is off (`tolerance: 0`), so every run takes its full budget unless it diverges.
 */
export function descend(f: ObjectiveFn, start: Vec, lr: number, steps: number, beta = 0): Trajectory {
  const options = { stepSize: lr, tolerance: 0, divergeAbove: DIVERGED }
  const alg = beta === 0 ? gradientDescent(f, options) : momentum(f, { ...options, momentum: beta })
  const run = trace(alg, { x0: start }, steps, {
    record: { x: (s) => s.x, value: (s) => s.value },
    timing: false,
  })
  const xs = dense.data(run.series.x)
  const path = Array.from(run.index, (_, i): Vec => [xs[2 * i], xs[2 * i + 1]])
  return { path, losses: Array.from(dense.data(run.series.value)), diverged: run.meta.stopped === 'diverged' }
}

/** L − L*, floored so that a log axis stays finite once a run has converged. */
export const excess = (losses: number[], optimum: number) => losses.map((l) => Math.max(l - optimum, 1e-12))

/** Round a clicked grid coordinate to the slider precision. */
export const snap = (v: number) => Math.round(v * 100) / 100

/** Snap a dragged coordinate and keep it within the plotted grid's axis values. */
export const inGrid = (v: number, axis: number[]) => snap(Math.min(Math.max(v, axis[0]), axis[axis.length - 1]))

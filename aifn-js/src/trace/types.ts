/**
 * The trace protocol's types. See `aifn-js/README.md` (Traces) and `docs/aifn-plan.md` §4. Streams come from
 * `aifn/random` and recorded series are `aifn/tensor` tensors.
 */

import type { Stream } from 'aifn/random'
import type { Tensor } from 'aifn/tensor'

/**
 * An iterative algorithm as a pure description. `step` must not mutate its argument: it returns the next state.
 * A state that sets `diverged: true` stops the runners with `stopped: 'diverged'`.
 */
export interface Algorithm<Opts, State> {
  name: string
  /** The initial state (step 0). Randomness comes only from `s`, so the same stream gives the same states. */
  init(opts: Opts, s?: Stream): State
  /** The next state; pure. */
  step(state: State): State
  /** True once the algorithm has converged or otherwise finished; the runners stop there. */
  done?(state: State): boolean
}

/**
 * What a recorder may return: a number, a flat array, a nested array (rectangular), or a `Tensor` (any strides or
 * dtype; recorded as float64).
 */
export type Recorded = number | ArrayLike<number> | readonly Recorded[] | Tensor

/** Maps a state to a recorded quantity; runs on kept steps only. */
export type Recorder<State> = (state: State, step: number) => Recorded

/** Why a run stopped: the algorithm's `done`, the step limit, or divergence (a flag or a non-finite recording). */
export type StopReason = 'done' | 'limit' | 'diverged'

/** How to run a trace. */
export interface TraceOptions<State> {
  /** Keep every `every`-th step (step numbers divisible by `every`), plus the final step. Default 1. */
  every?: number
  /** Named recorders; each becomes a series stacked over kept steps. */
  record?: Record<string, Recorder<State>>
  /** Also store the state at every multiple of this many steps, for fast `seek`. Default: none. */
  checkpointEvery?: number
  /** The stream passed to `init`. */
  stream?: Stream
  /**
   * Stop with `stopped: 'diverged'` when a recording is infinite, or NaN after that series has been finite. A NaN
   * before a series has had any finite value means "not defined yet" (e.g. a step size at step 0). Default true.
   */
  stopOnNonFinite?: boolean
}

/** Stored states for `seek`, keyed by step number (ascending). */
export interface Checkpoints<State> {
  index: number[]
  states: State[]
}

/** Timing of a traced run. All times are in milliseconds from `performance.now()` (or `Date.now()` without it). */
export interface TraceTiming {
  /** Time spent in `step` for each computed step: entry `t` is the step from state t to state t + 1. */
  stepMs: Float64Array
  /** Wall-clock time since the run began at each kept step (aligned with `index`); non-decreasing. */
  elapsedMs: Float64Array
  /** Total time spent in `step` (the sum of `stepMs`). */
  totalMs: number
  /** Steps per second of step time: `steps / (totalMs / 1000)`; `Infinity` when too fast to measure. */
  perSecond: number
  /** Time per named phase: `init`, `record` (the recorders), and every `profile(name, fn)` called inside a step. */
  phases: Record<string, number>
}

/** A traced run: the kept states, recorded series, timing and metadata. */
export interface Trace<State> {
  /** Kept states; `steps[0]` is the initial state and the last entry is always the final state. */
  steps: State[]
  /** The step number of each kept state. */
  index: number[]
  /**
   * Each recorded quantity stacked over kept steps: a contiguous float64 tensor of shape
   * `[steps.length, ...valueShape]`.
   */
  series: Record<string, Tensor>
  timing: TraceTiming
  meta: {
    algorithm: string
    stopped: StopReason
    /** The number of steps computed (the step number of the final state). */
    steps: number
    every: number
    checkpointEvery: number | null
    opts: unknown
    /** The key of the stream passed to `init`, if any. */
    seed?: string
  }
  /** States stored every `checkpointEvery` steps (always including step 0). */
  checkpoints: Checkpoints<State>
}

import type { Stream } from 'aifn/random'
import type { Tensor } from 'aifn/tensor'
import { flattenRecorded, makeSeries, seriesData } from './series'
import type { Algorithm, Checkpoints, Recorder, StopReason, Trace, TraceOptions } from './types'

/** A monotone clock in milliseconds: `performance.now()` where it exists (browsers, workers, Node), else `Date.now()`. */
export const now: () => number =
  typeof globalThis.performance?.now === 'function' ? () => globalThis.performance.now() : () => Date.now()

// ---------------------------------------------------------------------------------------------------------------------
// Profiling. A step runs synchronously, so a module-level "current phases" record is enough: the runner sets it around
// each `step` call and `profile` adds to it. Outside a traced step, `profile` just calls `fn`.

let currentPhases: Record<string, number> | null = null

/**
 * Times `fn` as the named phase of the current step: inside a traced step it adds the elapsed milliseconds to
 * `trace.timing.phases[name]`; elsewhere it only calls `fn`. Nested phases each count their own inclusive time.
 */
export function profile<T>(name: string, fn: () => T): T {
  const phases = currentPhases
  if (!phases) return fn()
  const start = now()
  try {
    return fn()
  } finally {
    phases[name] = (phases[name] ?? 0) + (now() - start)
  }
}

function timedStep<S>(alg: Algorithm<unknown, S>, state: S, phases: Record<string, number>): [S, number] {
  const previous = currentPhases
  currentPhases = phases
  const start = now()
  try {
    const next = alg.step(state)
    return [next, now() - start]
  } finally {
    currentPhases = previous
  }
}

/** True when a state carries the flag `diverged: true`. */
const flaggedDiverged = (state: unknown) =>
  typeof state === 'object' && state !== null && (state as { diverged?: unknown }).diverged === true

/** Why stepping must stop at this state, if it must: a divergence flag first, then the algorithm's `done`. */
function stopReason<S>(alg: Algorithm<never, S>, state: S): StopReason | null {
  if (flaggedDiverged(state)) return 'diverged'
  if (alg.done?.(state)) return 'done'
  return null
}

// ---------------------------------------------------------------------------------------------------------------------
// Runners without history.

/**
 * Runs `alg` for at most `n` steps and returns the final state. Stops early when `done(state)` is true or the state
 * is flagged `diverged`, so `run(alg, opts, n)` is the state a trace of `n` steps ends on.
 */
export function run<Opts, S>(alg: Algorithm<Opts, S>, opts: Opts, n: number, options: { stream?: Stream } = {}): S {
  let state = alg.init(opts, options.stream)
  for (let t = 0; t < n && !stopReason(alg, state); t++) state = alg.step(state)
  return state
}

/**
 * The state at step `i`, equal to `run(alg, opts, i)`. With `checkpoints` (a trace, or stored checkpoints), it starts
 * from the latest stored state at or before step `i` rather than from `init`, so scrubbing a long run is cheap.
 */
export function seek<Opts, S>(
  alg: Algorithm<Opts, S>,
  opts: Opts,
  i: number,
  options: { checkpoints?: Trace<S> | Checkpoints<S>; stream?: Stream } = {},
): S {
  let t = 0
  let state: S | undefined
  const sources: Checkpoints<S>[] = []
  const c = options.checkpoints
  if (c) {
    if ('checkpoints' in c) sources.push(c.checkpoints, { index: c.index, states: c.steps })
    else sources.push(c)
  }
  for (const source of sources) {
    // Indices are ascending: take the last one at or before i.
    for (let k = source.index.length - 1; k >= 0; k--) {
      if (source.index[k] <= i) {
        if (state === undefined || source.index[k] > t) {
          t = source.index[k]
          state = source.states[k]
        }
        break
      }
    }
  }
  if (state === undefined) state = alg.init(opts, options.stream)
  for (; t < i && !stopReason(alg, state); t++) state = alg.step(state)
  return state
}

/**
 * A generator of `{ step, state }` for play loops, starting at step 0. It ends after yielding a state that is done or
 * flagged diverged (with `stopped` set); otherwise it runs for as long as it is pulled.
 */
export function* live<Opts, S>(
  alg: Algorithm<Opts, S>,
  opts: Opts,
  options: { stream?: Stream } = {},
): Generator<{ step: number; state: S; stopped?: StopReason }, void, unknown> {
  let state = alg.init(opts, options.stream)
  for (let step = 0; ; step++) {
    const stopped = stopReason(alg, state)
    if (stopped) {
      yield { step, state, stopped }
      return
    }
    yield { step, state }
    state = alg.step(state)
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// The trace builder: one incremental engine behind `trace`, `extend` and `timeSliced`.

type Column = { valueShape: number[] | null; values: number[]; finiteSeen?: boolean }

/** Options the trace was made with, so `extend` can reuse its recorders. Functions are not part of the plain trace. */
const madeWith = new WeakMap<object, TraceOptions<never>>()

interface Builder<S> {
  /** Step until step `target`, a stop, or the clock passes `deadline` (at least one step per call). */
  advance(target: number, deadline?: number): void
  readonly t: number
  readonly stopped: StopReason | null
  /** The trace so far; the current state is included as the final kept step. Does not change the builder. */
  snapshot(): Trace<S>
}

function createBuilder<S>(
  alg: Algorithm<unknown, S>,
  options: TraceOptions<S>,
  start: {
    state: S
    t: number
    opts: unknown
    seed?: string
    // Accumulators to continue from (for extend).
    steps?: S[]
    index?: number[]
    columns?: Map<string, Column>
    stepMs?: number[]
    elapsed?: number[]
    phases?: Record<string, number>
    checkpoints?: Checkpoints<S>
    elapsedOffset?: number
  },
): Builder<S> {
  const every = Math.max(1, Math.floor(options.every ?? 1))
  const checkpointEvery = options.checkpointEvery ? Math.max(1, Math.floor(options.checkpointEvery)) : null
  const recorders = Object.entries(options.record ?? {}) as [string, Recorder<S>][]
  const stopOnNonFinite = options.stopOnNonFinite ?? true

  let state = start.state
  let t = start.t
  const steps = start.steps ?? []
  const index = start.index ?? []
  const columns =
    start.columns ?? new Map<string, Column>(recorders.map(([name]) => [name, { valueShape: null, values: [] }]))
  const stepMs = start.stepMs ?? []
  const elapsed = start.elapsed ?? []
  const phases = start.phases ?? {}
  const checkpoints = start.checkpoints ?? { index: [], states: [] }
  const began = now() - (start.elapsedOffset ?? 0)
  let stopped: StopReason | null = null

  /** Runs the recorders on a state; returns one row per series and whether any value was not finite. */
  function recordRow(s: S, step: number): { rows: [string, number[]][]; nonFinite: boolean } {
    const startRecord = now()
    let nonFinite = false
    const rows: [string, number[]][] = []
    for (const [name, recorder] of recorders) {
      const { shape, values } = flattenRecorded(recorder(s, step), name)
      const column = columns.get(name)!
      if (
        column.valueShape &&
        (column.valueShape.length !== shape.length || column.valueShape.some((d, i) => d !== shape[i]))
      )
        throw new Error(
          `trace: recorder "${name}" changed shape from [${column.valueShape}] to [${shape}] at step ${step}`,
        )
      // Divergence: an infinity anywhere, or a NaN in a series that has already been finite. A NaN before a series has
      // had any finite value means "not defined yet" (a step size or trial before the first step), not divergence.
      if (!nonFinite && values.some((v) => v === Infinity || v === -Infinity || (Number.isNaN(v) && column.finiteSeen)))
        nonFinite = true
      if (!column.finiteSeen && values.some(Number.isFinite)) column.finiteSeen = true
      rows.push([name, values])
      if (!column.valueShape) column.valueShape = shape
    }
    phases.record = (phases.record ?? 0) + (now() - startRecord)
    return { rows, nonFinite }
  }

  function keep(s: S, step: number): boolean {
    const { rows, nonFinite } = recordRow(s, step)
    steps.push(s)
    index.push(step)
    elapsed.push(now() - began)
    for (const [name, values] of rows) columns.get(name)!.values.push(...values)
    return nonFinite && stopOnNonFinite
  }

  // A fresh start keeps step 0 and checkpoints it.
  if (steps.length === 0) {
    checkpoints.index.push(t)
    checkpoints.states.push(state)
    if (keep(state, t)) stopped = 'diverged'
  }
  stopped ??= stopReason(alg as Algorithm<never, S>, state)

  return {
    get t() {
      return t
    },
    get stopped() {
      return stopped
    },
    advance(target, deadline) {
      let first = true
      while (t < target && !stopped) {
        if (!first && deadline !== undefined && now() >= deadline) return
        first = false
        const [next, ms] = timedStep(alg, state, phases)
        state = next
        t++
        stepMs.push(ms)
        if (checkpointEvery && t % checkpointEvery === 0) {
          checkpoints.index.push(t)
          checkpoints.states.push(state)
        }
        if (t % every === 0 && keep(state, t)) stopped = 'diverged'
        stopped ??= stopReason(alg as Algorithm<never, S>, state)
      }
    },
    snapshot() {
      // The final state is always kept; when it is off the `every` grid it is an extra row, added here without
      // changing the accumulators, so extending a trace later leaves exactly the rows a longer run would keep.
      let extra: { rows: [string, number[]][]; elapsed: number } | null = null
      let reason: StopReason = stopped ?? 'limit'
      if (index[index.length - 1] !== t) {
        const { rows, nonFinite } = recordRow(state, t)
        extra = { rows, elapsed: now() - began }
        if (nonFinite && stopOnNonFinite) reason = 'diverged'
      }
      const kept = index.length + (extra ? 1 : 0)
      const series: Record<string, Tensor> = {}
      for (const [name, column] of columns) {
        const extraValues = extra?.rows.find(([n]) => n === name)?.[1] ?? []
        const data = new Float64Array(column.values.length + extraValues.length)
        data.set(column.values)
        data.set(extraValues, column.values.length)
        series[name] = makeSeries([kept, ...(column.valueShape ?? [])], data)
      }
      const totalMs = stepMs.reduce((a, b) => a + b, 0)
      const trace: Trace<S> = {
        steps: extra ? [...steps, state] : [...steps],
        index: extra ? [...index, t] : [...index],
        series,
        timing: {
          stepMs: Float64Array.from(stepMs),
          elapsedMs: Float64Array.from(extra ? [...elapsed, extra.elapsed] : elapsed),
          totalMs,
          perSecond: t === 0 ? 0 : totalMs > 0 ? t / (totalMs / 1000) : Infinity,
          phases: { ...phases },
        },
        meta: { algorithm: alg.name, stopped: reason, steps: t, every, checkpointEvery, opts: start.opts },
        checkpoints: { index: [...checkpoints.index], states: [...checkpoints.states] },
      }
      if (start.seed !== undefined) trace.meta.seed = start.seed
      madeWith.set(trace, options as TraceOptions<never>)
      return trace
    },
  }
}

function freshBuilder<Opts, S>(alg: Algorithm<Opts, S>, opts: Opts, options: TraceOptions<S>): Builder<S> {
  const phases: Record<string, number> = {}
  const startInit = now()
  const state = alg.init(opts, options.stream)
  phases.init = now() - startInit
  return createBuilder(alg as Algorithm<unknown, S>, options, {
    state,
    t: 0,
    opts,
    seed: options.stream?.key,
    phases,
  })
}

/**
 * Runs `alg` for at most `n` steps and returns its trace: the states at steps divisible by `every` (and always the
 * final state), each recorder's values stacked over those kept steps, per-step timing and why it stopped (`done`,
 * `limit`, or `diverged` when a state is flagged `diverged` or a recording is not finite).
 */
export function trace<Opts, S>(
  alg: Algorithm<Opts, S>,
  opts: Opts,
  n: number,
  options: TraceOptions<S> = {},
): Trace<S> {
  const builder = freshBuilder(alg, opts, options)
  builder.advance(n)
  return builder.snapshot()
}

/**
 * Continues a trace by `m` more steps from its final state, as though it had been traced for `meta.steps + m` steps
 * in the first place (same kept steps, series and checkpoints). A trace that stopped `done` or `diverged` is returned
 * unchanged. `options` defaults to those the trace was made with (its recorders are remembered); `every` and
 * `checkpointEvery` always come from the trace. `opts` is stored in `meta.opts`.
 */
export function extend<Opts, S>(
  previous: Trace<S>,
  alg: Algorithm<Opts, S>,
  opts: Opts,
  m: number,
  options?: TraceOptions<S>,
): Trace<S> {
  if (previous.meta.stopped !== 'limit' || m <= 0) return previous
  const base = (options ?? (madeWith.get(previous) as TraceOptions<S> | undefined) ?? {}) as TraceOptions<S>
  const merged: TraceOptions<S> = {
    ...base,
    every: previous.meta.every,
    checkpointEvery: previous.meta.checkpointEvery ?? undefined,
  }
  const names = Object.keys(merged.record ?? {})
  const recorded = Object.keys(previous.series)
  if (names.length !== recorded.length || names.some((n) => !(n in previous.series)))
    throw new Error(`trace: extend needs the same recorders as the trace (${recorded.join(', ') || 'none'})`)

  // Drop the extra final row (the final state kept off the `every` grid); a longer run would not keep it.
  const last = previous.index.length - 1
  const dropLast = previous.index[last] % previous.meta.every !== 0 ? 1 : 0
  const keptRows = previous.index.length - dropLast
  const columns = new Map<string, Column>()
  for (const name of recorded) {
    const s = previous.series[name]
    const valueShape = s.shape.slice(1)
    const width = valueShape.reduce((a, b) => a * b, 1)
    const values = Array.from(seriesData(s).subarray(0, keptRows * width))
    columns.set(name, { valueShape, values, finiteSeen: values.some(Number.isFinite) })
  }
  const elapsed = Array.from(previous.timing.elapsedMs.subarray(0, keptRows))
  const builder = createBuilder(alg as Algorithm<unknown, S>, merged, {
    state: previous.steps[last],
    t: previous.meta.steps,
    opts,
    seed: previous.meta.seed,
    steps: previous.steps.slice(0, keptRows),
    index: previous.index.slice(0, keptRows),
    columns,
    stepMs: Array.from(previous.timing.stepMs),
    elapsed,
    phases: { ...previous.timing.phases },
    checkpoints: { index: [...previous.checkpoints.index], states: [...previous.checkpoints.states] },
    elapsedOffset: previous.timing.elapsedMs[previous.timing.elapsedMs.length - 1] ?? 0,
  })
  builder.advance(previous.meta.steps + m)
  return builder.snapshot()
}

/**
 * Runs a trace in slices of about `budgetMs` of work each, yielding the partial trace after every slice and the full
 * trace last. Between slices it waits on `schedule(resume)`, which the caller supplies (for example
 * `requestAnimationFrame` in a page); the default is `setTimeout(resume, 0)`. Stop early by breaking out of the loop.
 * Each partial trace is a valid trace of the steps so far (`meta.stopped` is `limit` until the run ends).
 */
export async function* timeSliced<Opts, S>(
  alg: Algorithm<Opts, S>,
  opts: Opts,
  n: number,
  budgetMs: number,
  options: TraceOptions<S> & { schedule?: (resume: () => void) => void } = {},
): AsyncGenerator<Trace<S>, void, unknown> {
  const schedule = options.schedule ?? ((resume: () => void) => void setTimeout(resume, 0))
  const builder = freshBuilder(alg, opts, options)
  for (;;) {
    builder.advance(n, now() + budgetMs)
    yield builder.snapshot()
    if (builder.t >= n || builder.stopped) return
    await new Promise<void>((resume) => schedule(resume))
  }
}

/**
 * A copy of a trace thinned to at most `maxPoints` kept steps, evenly spaced over the kept steps and always keeping
 * the first and last, for drawing. Series, `index` and `elapsedMs` are thinned together; `stepMs` and the checkpoints
 * are unchanged. Do not `extend` a decimated trace.
 */
export function decimate<S>(t: Trace<S>, maxPoints: number): Trace<S> {
  const kept = t.index.length
  if (kept <= maxPoints || maxPoints < 2) return t
  const picks: number[] = []
  for (let k = 0; k < maxPoints; k++) {
    const p = Math.round((k * (kept - 1)) / (maxPoints - 1))
    if (p !== picks[picks.length - 1]) picks.push(p)
  }
  const series: Record<string, Tensor> = {}
  for (const [name, s] of Object.entries(t.series)) {
    const valueShape = s.shape.slice(1)
    const width = valueShape.reduce((a, b) => a * b, 1)
    const source = seriesData(s)
    const data = new Float64Array(picks.length * width)
    picks.forEach((p, r) => data.set(source.subarray(p * width, (p + 1) * width), r * width))
    series[name] = makeSeries([picks.length, ...valueShape], data)
  }
  return {
    ...t,
    steps: picks.map((p) => t.steps[p]),
    index: picks.map((p) => t.index[p]),
    series,
    timing: { ...t.timing, elapsedMs: Float64Array.from(picks, (p) => t.timing.elapsedMs[p]) },
  }
}

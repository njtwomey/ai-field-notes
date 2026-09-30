/**
 * Traces: the protocol every iterative algorithm in aifn follows, and the runners that drive it.
 *
 * An `Algorithm` is a pure description (`init`, `step`, optional `done`). `run` returns the final state; `trace`
 * returns the kept states with recorded series, timing and metadata; `seek` returns the state at one step (from
 * checkpoints when given a trace); `extend` continues a trace; `live` is a generator for play loops; `timeSliced`
 * yields partial traces for interfaces; `decimate` thins a trace for drawing; `profile` times named phases in a step.
 * See `aifn-js/README.md` (Traces) and `docs/aifn-plan.md` §4.
 */
export type { Algorithm, Checkpoints, Recorded, Recorder, StopReason, Trace, TraceOptions, TraceTiming } from './types'
export { decimate, extend, live, now, profile, run, seek, timeSliced, trace } from './runners'
export { seriesComponents } from './series'

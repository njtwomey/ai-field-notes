/** Running several chains of one sampler on child streams, and stacking their draws for the diagnostics. */

import { replicate, type Stream } from 'aifn/random'
import { fromData, type Tensor, type Vector } from 'aifn/tensor'
import { trace, type Algorithm, type Recorder, type Trace } from 'aifn/trace'
import { data } from './util'

/** Options for `sampleChains`. */
export type SampleChainsOptions<S> = {
  /** Number of chains m. Default 4. */
  chains?: number
  /** Steps per chain. */
  steps: number
  /** Chain k runs on `stream.child(k)`. */
  stream: Stream
  /** Steps discarded from the start of each chain (burn-in). Default 0. */
  warmup?: number
  /** Keep every `every`-th state (thinning). Default 1. */
  every?: number
  /** Extra recorders for each chain's trace (`x` is always recorded). */
  record?: Record<string, Recorder<S>>
}

/** The chains' traces and their draws stacked for the diagnostics. */
export type ChainsResult<S> = {
  traces: Trace<S>[]
  /** The kept draws after warmup, m×n×d: chain, draw, coordinate (the initial state is never a draw). */
  draws: Tensor
  /** The step number of each draw (length n). */
  steps: number[]
}

/**
 * Run m chains of `alg`, chain k from `start(k)` on `stream.child(k)` (so chain k's draws depend only on the stream
 * and k: adding chains keeps the first ones), and stack the draws x after `warmup` into an m×n×d tensor for
 * `effectiveSampleSize`, `splitRhat` and `monteCarloStandardError`.
 */
export function sampleChains<Opts, S extends { x: Vector }>(
  alg: Algorithm<Opts, S>,
  start: Opts | ((k: number) => Opts),
  options: SampleChainsOptions<S>,
): ChainsResult<S> {
  const { chains = 4, steps, stream, warmup = 0, every = 1 } = options
  const optsFor = (k: number) => (typeof start === 'function' ? (start as (k: number) => Opts)(k) : start)
  const traces = replicate(
    chains,
    stream,
    (s, k) => trace(alg, optsFor(k), steps, { stream: s, every, record: { x: (st: S) => st.x, ...options.record } }),
    { cache: false },
  )
  const keep = traces[0].index.map((i, j) => [i, j] as const).filter(([i]) => i > warmup && i > 0)
  const n = Math.min(...traces.map((tr) => tr.index.filter((i) => i > warmup && i > 0).length))
  const d = traces[0].steps[0].x.shape[0]
  const out = new Float64Array(chains * n * d)
  traces.forEach((tr, k) => {
    const x = data(tr.series.x)
    const rows = tr.index.map((i, j) => [i, j] as const).filter(([i]) => i > warmup && i > 0)
    for (let r = 0; r < n; r++) out.set(x.subarray(rows[r][1] * d, (rows[r][1] + 1) * d), (k * n + r) * d)
  })
  return { traces, draws: fromData(out, [chains, n, d]), steps: keep.slice(0, n).map(([i]) => i) }
}

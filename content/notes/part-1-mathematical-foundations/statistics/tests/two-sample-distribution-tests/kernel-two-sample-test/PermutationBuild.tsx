import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Histogram,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, permutation, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { histogram } from 'aifn-compute/probability/stats'
import { kernelMatrix, medianHeuristic, mmd2, twoDimensionalPair } from '../_shared/twoSample'

const BOX = 4
const RESAMPLES = 400
const BINS = 30
const clamp = (v: number) => Math.min(BOX - 0.5, Math.max(-BOX + 0.5, v))

/**
 * The kernel two-sample test built one permutation at a time: each step shuffles the pooled sample's labels, splits it
 * into groups of the original sizes, and adds that split's MMD² to the null histogram. The p-value is the share of
 * splits at least as extreme as the observed one, with the +1 correction that the engine's `permutationTest` uses.
 */
export function PermutationBuild() {
  const state = useFigureState({
    mx: float(0.4, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₁', onChart: true }),
    my: float(0, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₂', onChart: true }),
    rho: float(0, { min: -0.95, max: 0.95, step: 0.05, label: 'correlation of Q, ρ' }),
    median: setting(true, { label: 'kernel width by the median heuristic' }),
    logSigma: float(0, { min: -1, max: 1, step: 0.02, label: 'log₁₀ kernel width σ' }),
    n: int(60, { min: 10, max: 150, label: 'points in each sample', suggestions: [20, 60, 150] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => twoDimensionalPair(state.seed, state.n, state.n, [state.mx, state.my], state.rho),
    [state.seed, state.n, state.mx, state.my, state.rho],
  )

  const run = useComputed(
    () => {
      const pooled = [...data.x, ...data.y]
      const N = pooled.length
      const n = data.x.length
      const sigma = state.median ? medianHeuristic(pooled) : 10 ** state.logSigma
      const K = kernelMatrix(pooled, sigma)
      const observed = mmd2(
        K,
        N,
        Array.from({ length: n }, (_, i) => i),
        Array.from({ length: N - n }, (_, i) => n + i),
      )
      // Each permutation's first n indices form the relabelled first sample.
      const orders = Array.from({ length: RESAMPLES }, (_, r) =>
        Int32Array.from(toFlat(permutation(child(stream(state.seed), 'permutation', r), N))),
      )
      const nulls = orders.map((o) => mmd2(K, N, o.subarray(0, n), o.subarray(n)))
      // One bin range for every step, and the final histogram's tallest bin as the height, so the bars grow into a
      // fixed chart as shuffles accumulate.
      const lo = Math.min(...nulls, observed)
      const hi = Math.max(...nulls, observed)
      const pad = 0.05 * (hi - lo)
      const binRange: [number, number] = [lo - pad, hi + pad]
      const top = 1.1 * Math.max(...toFlat(histogram(nulls, { bins: BINS, range: binRange }).counts))
      return { pooled, n, sigma, observed, orders, nulls, binRange, top }
    },
    [data, state.median, state.logSigma, state.seed],
    { mode: 'release' },
  )
  const r = run.value

  // The walk-through restarts at step 0 whenever the run changes.
  const [position, setPosition] = useState({ r, k: 0 })
  const k = position.r === r ? position.k : 0

  // Step 0 shows the observed labels; step k ≥ 1 shows the k-th relabelling and the first k null values.
  const firstGroup = useMemo(() => {
    const inFirst = new Uint8Array(r.pooled.length)
    if (k === 0) inFirst.fill(1, 0, r.n)
    else for (const i of r.orders[k - 1].subarray(0, r.n)) inFirst[i] = 1
    return inFirst
  }, [r, k])
  const seen = r.nulls.slice(0, k)
  const extreme = seen.filter((v) => v >= r.observed).length
  const pValue = (1 + extreme) / (1 + k)
  const pts = (keep: 0 | 1) => r.pooled.filter((_, i) => firstGroup[i] === keep)
  const xa = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const ya = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: xa })
  const xb = useAxis({ label: 'MMD²', range: r.binRange })
  const yb = useAxis({ label: 'shuffles', range: [0, r.top] })
  return (
    <Figure
      title="Building the permutation null"
      state={state}
      caption="Left: the pooled points, labelled as the first or second sample. At step 0 the labels are the real ones; each later step shuffles them, as the null hypothesis allows when both samples come from one distribution. Right: the MMD² of every shuffle so far, against the observed MMD² (black line). Play to watch the null fill in and the p-value settle. Drag the marker to move Q's mean: near P's mean the observed value sits inside the null, far from it the observed value sits beyond every shuffle."
      controls={
        <Player value={k} onChange={(v) => setPosition({ r, k: v })} count={RESAMPLES + 1} label="permutation" />
      }
      readouts={
        <>
          <Readout label="permutations" value={k} />
          <Readout label="observed MMD²" value={formatNumber(r.observed)} />
          <Readout label="this shuffle's MMD²" value={k ? formatNumber(r.nulls[k - 1]) : '—'} />
          <Readout label="shuffles at least as large" value={`${extreme} of ${k}`} />
          <Readout label="p-value (1 + b)/(1 + k)" value={formatNumber(pValue)} />
          <Readout label="σ" value={formatNumber(r.sigma)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          <Points name="labelled first sample" x={pts(1).map((p) => p[0])} y={pts(1).map((p) => p[1])} slot={0} />
          <Points name="labelled second sample" x={pts(0).map((p) => p[0])} y={pts(0).map((p) => p[1])} slot={1} />
          <Handle
            kind="point"
            at={[state.mx, state.my]}
            label="mean of Q"
            onDrag={([x, y]) => {
              state.set('mx', clamp(x))
              state.set('my', clamp(y))
            }}
          />
        </Plot>
        <Plot x={xb} y={yb}>
          {seen.length > 1 && (
            <Histogram
              name="MMD² of the shuffles"
              values={seen}
              bins={BINS}
              range={r.binRange}
              normalize="count"
              slot={2}
              stale={run.stale}
            />
          )}
          <Segments
            name="observed MMD²"
            segments={[{ from: [r.observed, 0], to: [r.observed, r.top] }]}
            emphasis
            width={2}
          />
        </Plot>
      </div>
    </Figure>
  )
}

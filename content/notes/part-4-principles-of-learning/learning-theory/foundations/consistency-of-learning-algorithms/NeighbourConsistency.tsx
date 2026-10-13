import { type BiasVarianceResult } from 'aifn-methods/theory/bias-variance'
import {
  call,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  type Task,
} from 'aifn-render'

const fmt = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—'
const SIZES = [10, 20, 40, 80, 160, 320]
const REPEATS = 20
type Totals = BiasVarianceResult['totals']
type Rule = { name: string; k: (m: number) => number; slot: number }

/**
 * k-nearest-neighbour regression of y = sin(πx) + σε, its expected test error against the sample size m for three
 * rules: k = 1, a fixed k, and k growing like √m. Only the last satisfies k → ∞ and k/m → 0.
 */
export function NeighbourConsistency() {
  const state = useFigureState({
    noise: slider(0.1, 1, 0.5, { step: 0.05, label: 'noise σ' }),
    k: int(10, { min: 2, max: 50, suggestions: [5, 10, 30], label: 'fixed k' }),
  })
  const rules: Rule[] = [
    { name: 'k = 1', k: () => 1, slot: 0 },
    { name: `k = ${state.k} (fixed)`, k: (m) => Math.min(state.k, m), slot: 1 },
    { name: 'k = √m', k: (m) => Math.max(1, Math.round(Math.sqrt(m))), slot: 2 },
  ]
  // Every (rule, m) study runs in the compute worker, which evaluates an array of calls as one task.
  const run = useComputed(
    () =>
      rules.flatMap((rule) =>
        SIZES.map((m) =>
          call<BiasVarianceResult>(
            'theory/bias-variance/biasVariance',
            call('foundation/random/stream', 'consistency/knn'),
            {
              target: 'sine',
              noise: state.noise,
              n: m,
              repeats: REPEATS,
              grid: 51,
              keep: 0,
              model: { kind: 'knn', k: rule.k(m) },
            },
          ),
        ),
      ) as unknown as Task<BiasVarianceResult[]>,
    [state.noise, state.k],
    {
      mode: 'worker',
      initial: [] as Totals[],
      then: (results) => results.map((r) => r.totals),
    },
  )
  const totals = run.value
  const ready = totals.length === rules.length * SIZES.length
  const bayes = state.noise ** 2
  const errorOf = (r: number) => (ready ? SIZES.map((_, i) => totals[r * SIZES.length + i].error) : [])
  const last = (r: number) => (ready ? totals[r * SIZES.length + SIZES.length - 1].error : undefined)

  const mAxis = useAxis({
    label: 'training points m',
    range: [10, 320],
    log: true,
    format: (v) => String(Math.round(v)),
  })
  const eAxis = useAxis({ label: 'expected squared error', hold: 'union', key: String(state.noise) })
  return (
    <Figure
      title="Which nearest-neighbour rules are consistent"
      state={state}
      caption="k-nearest-neighbour regression on y = sin(πx) + σε, x uniform on [−1, 1], with the expected squared error on fresh data averaged over 20 training sets at each m. The Bayes risk is σ² (dark dashed). With k = 1 the error levels off near 2σ² (grey dashed): the nearest neighbour comes ever closer, but its label still carries its own noise. A fixed k levels off near σ²(1 + 1/k). Only k growing like √m, so that k → ∞ while k/m → 0, keeps closing the gap to σ². Computed in the background; the curves appear when the run finishes."
      readouts={
        <>
          <Readout label="Bayes risk σ²" value={fmt(bayes)} />
          <Readout label="k = 1 at m = 320" value={fmt(last(0))} />
          <Readout label={`k = ${state.k} at m = 320`} value={fmt(last(1))} />
          <Readout label="k = √m at m = 320" value={fmt(last(2))} />
        </>
      }
    >
      <Plot x={mAxis} y={eAxis} height={340}>
        <Curve name="Bayes risk σ²" x={[10, 320]} y={[bayes, bayes]} emphasis dashed />
        <Curve name="2σ²" x={[10, 320]} y={[2 * bayes, 2 * bayes]} muted dashed />
        {ready &&
          rules.map((rule, r) => (
            <Curve
              key={rule.slot}
              name={rule.name}
              slot={rule.slot}
              x={SIZES}
              y={errorOf(r)}
              showPoints
              stale={run.stale}
            />
          ))}
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

/*
 * A recommender with C = 4 contexts (viewed products, drawn uniformly) and K = 6 candidate items. Each (context, item)
 * cell has a true click rate MU. The production policy is a softmax over item popularity only, with a skew β; the new
 * policy is a softmax over the true click rates, so it recommends different items in each context.
 *
 * Every estimator is a function of the per-cell impression counts N and click counts S of a log, so the simulation
 * stores only those. Changing the clip threshold or the reward model then costs a few hundred operations per log.
 */
const MU = [
  [0.1, 0.08, 0.3, 0.05, 0.12, 0.04],
  [0.12, 0.1, 0.06, 0.28, 0.05, 0.08],
  [0.09, 0.11, 0.07, 0.06, 0.25, 0.1],
  [0.11, 0.09, 0.05, 0.07, 0.06, 0.32],
]
const C = MU.length
const K = MU[0].length
const POPULARITY = [1, 0.8, 0.3, 0.2, 0.1, 0]
const REPS = 200
const SIZES = ['200', '1000', '5000'] as const
type Size = (typeof SIZES)[number]
const M_MAX = 100
const M_GRID = Array.from({ length: 41 }, (_, i) => 1 + (i * (M_MAX - 1)) / 40)

function softmax(s: number[]): number[] {
  const top = Math.max(...s)
  const e = s.map((v) => Math.exp(v - top))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

const TARGET = MU.map((row) => softmax(row.map((m) => 15 * m)))
const TRUE_VALUE = TARGET.reduce((acc, p, x) => acc + p.reduce((s, q, a) => s + q * MU[x][a], 0), 0) / C
/** Ignores the context: each item's click rate averaged over contexts. */
const WRONG = MU[0].map((_, a) => MU.reduce((s, row) => s + row[a], 0) / C)

type Log = { N: number[][]; S: number[][] }

function simulate(n: number, logging: number[], seed: number): Log[] {
  const cum = logging.map((_, a) => logging.slice(0, a + 1).reduce((s, v) => s + v, 0))
  return Array.from({ length: REPS }, (_, j) => {
    const r = stream(seed * 7919 + j * 104729 + 17)
    const N = MU.map(() => new Array<number>(K).fill(0))
    const S = MU.map(() => new Array<number>(K).fill(0))
    for (let i = 0; i < n; i++) {
      const x = Math.floor(uniform(r) * C)
      const u = uniform(r)
      let a = 0
      while (a < K - 1 && u > cum[a]) a++
      N[x][a]++
      if (uniform(r) < MU[x][a]) S[x][a]++
    }
    return { N, S }
  })
}

type Key = 'ips' | 'cips' | 'snips' | 'dm' | 'dr'
const ESTIMATORS: { key: Key; label: string }[] = [
  { key: 'ips', label: 'IPS' },
  { key: 'cips', label: 'clipped IPS' },
  { key: 'snips', label: 'SNIPS' },
  { key: 'dm', label: 'direct method' },
  { key: 'dr', label: 'doubly robust' },
]

/** The five estimates from one log's cell counts, for weights w[x][a] and reward model muHat[x][a]. */
function estimates(log: Log, n: number, w: number[][], muHat: number[][], clip: number): Record<Key, number> {
  let ips = 0
  let cips = 0
  let sw = 0
  let dm = 0
  let correction = 0
  for (let x = 0; x < C; x++) {
    let nx = 0
    for (let a = 0; a < K; a++) {
      const N = log.N[x][a]
      const S = log.S[x][a]
      nx += N
      ips += w[x][a] * S
      cips += Math.min(w[x][a], clip) * S
      sw += w[x][a] * N
      correction += w[x][a] * (S - muHat[x][a] * N)
    }
    dm += nx * TARGET[x].reduce((s, p, a) => s + p * muHat[x][a], 0)
  }
  return { ips: ips / n, cips: cips / n, snips: sw > 0 ? ips / sw : 0, dm: dm / n, dr: (dm + correction) / n }
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const summary = (xs: number[]) => {
  const m = mean(xs)
  const sd = Math.sqrt(mean(xs.map((v) => (v - m) ** 2)))
  return { bias: m - TRUE_VALUE, sd, rmse: Math.sqrt((m - TRUE_VALUE) ** 2 + sd ** 2) }
}

/**
 * Sampling distributions of five counterfactual estimators of a new recommender's click rate, from simulated logs of a
 * production recommender, with a draggable clip threshold.
 */
export function EstimatorLab() {
  const state = useFigureState({
    skew: float(3, { min: 0, max: 5, step: 0.1, label: 'propensity skew β' }),
    misspec: float(1, { min: 0, max: 1, step: 0.05, label: 'reward-model misspecification' }),
    clip: slider(1, M_MAX, 10, { step: 0.5, label: 'clip threshold M' }),
    size: choice<Size>(
      SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString() })),
      '1000',
      { label: 'impressions per log n' },
    ),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const n = Number(state.size)

  const logging = useMemo(() => softmax(POPULARITY.map((s) => state.skew * s)), [state.skew])
  const weights = useMemo(() => TARGET.map((p) => p.map((q, a) => q / logging[a])), [logging])
  const logs = useMemo(() => simulate(n, logging, state.seed), [n, logging, state.seed])
  const muHat = useMemo(
    () => MU.map((row) => row.map((m, a) => (1 - state.misspec) * m + state.misspec * WRONG[a])),
    [state.misspec],
  )

  const reps = useMemo(
    () => logs.map((log) => estimates(log, n, weights, muHat, state.clip)),
    [logs, n, weights, muHat, state.clip],
  )
  // Clipped IPS over a grid of thresholds: only the clipped sum depends on M.
  const clipCurve = useMemo(
    () =>
      M_GRID.map((M) => {
        const xs = logs.map((log) => {
          let s = 0
          for (let x = 0; x < C; x++) for (let a = 0; a < K; a++) s += Math.min(weights[x][a], M) * log.S[x][a]
          return s / n
        })
        return summary(xs)
      }),
    [logs, weights, n],
  )

  const stats = Object.fromEntries(ESTIMATORS.map(({ key }) => [key, summary(reps.map((r) => r[key]))])) as Record<
    Key,
    ReturnType<typeof summary>
  >
  const productionCtr = logging.reduce((s, p, a) => s + p * mean(MU.map((row) => row[a])), 0)
  const maxWeight = Math.max(...weights.flat())

  const strip = [
    {
      name: 'estimates',
      x: ESTIMATORS.flatMap((_, i) => reps.map((_, j) => i + 1 + 0.6 * (((j * 0.618034) % 1) - 0.5))),
      y: ESTIMATORS.flatMap(({ key }) => reps.map((r) => r[key])),
      group: ESTIMATORS.flatMap((_, i) => reps.map(() => i)),
      groupNames: ESTIMATORS.map((e) => e.label),
    },
    { name: 'true value V(π)', x: [0.5, 5.5], y: [TRUE_VALUE, TRUE_VALUE], emphasis: true, dashed: true },
    {
      name: 'production CTR V(π₀)',
      x: [0.5, 5.5],
      y: [productionCtr, productionCtr],
      muted: true,
      dashed: true,
    },
  ] as const
  const tradeOff = [
    { name: 'CIPS RMSE', x: M_GRID, y: clipCurve.map((s) => s.rmse), slot: 1 },
    { name: 'CIPS |bias|', x: M_GRID, y: clipCurve.map((s) => Math.abs(s.bias)), slot: 5 },
    { name: 'CIPS sd', x: M_GRID, y: clipCurve.map((s) => s.sd), slot: 6 },
    { name: 'IPS RMSE', x: [1, M_MAX], y: [stats.ips.rmse, stats.ips.rmse], slot: 0, dashed: true },
    {
      name: 'SNIPS RMSE',
      x: [1, M_MAX],
      y: [stats.snips.rmse, stats.snips.rmse],
      slot: 2,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'estimator', range: [0.5, 5.5] })
  const yAxis = useAxis({ label: 'estimated click rate', hold: 'union' })
  const xAxis2 = useAxis({ label: 'clip threshold M', range: [1, M_MAX] })
  const yAxis2 = useAxis({ label: 'error', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Counterfactual estimator lab"
      state={state}
      caption="A production recommender shows one of six items on each of four product pages, with probabilities set by item popularity alone; the propensity skew β sharpens them. A new recommender, which follows the true click rates, is evaluated from 200 simulated logs. Top: the 200 estimates from each estimator against the true click rate of the new recommender (dashed ink) and the production click rate (dashed grey), which is what the log’s own average reports. Bottom: bias, spread and RMSE of clipped IPS as the clip threshold M varies; drag the guide to set M. The reward model used by the direct method and doubly robust blends the true click rates with a model that ignores the page; misspecification 1 is the page-blind model. Raise the skew and IPS spreads out; clip hard and the estimate falls below the truth; misspecify the model and the direct method is confidently wrong while doubly robust stays centred."

      readouts={
        <>
          <Readout label="true value V(π)" value={formatNumber(TRUE_VALUE)} />
          <Readout label="production CTR V(π₀)" value={formatNumber(productionCtr)} />
          <Readout
            label="smallest propensity, largest weight"
            value={`${formatNumber(Math.min(...logging))}, ${formatNumber(maxWeight)}`}
          />
          {ESTIMATORS.map(({ key, label }) => (
            <Readout
              key={key}
              label={`${label}: bias, sd, RMSE`}
              value={`${formatNumber(stats[key].bias)}, ${formatNumber(stats[key].sd)}, ${formatNumber(stats[key].rmse)}`}
            />
          ))}
        </>
      }
    >
      <div className="grid gap-4">
        <Plot x={xAxis} y={yAxis}>
          <Points {...strip[0]} />
          <Curve {...strip[1]} />
          <Curve {...strip[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...tradeOff[0]} />
          <Curve {...tradeOff[1]} />
          <Curve {...tradeOff[2]} />
          <Curve {...tradeOff[3]} />
          <Curve {...tradeOff[4]} />
          <Handle {...state.handle('clip', { label: 'M' })} />
        </Plot>
      </div>
    </Figure>
  )
}

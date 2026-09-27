import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

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
    const r = rng(seed * 7919 + j * 104729 + 17)
    const N = MU.map(() => new Array<number>(K).fill(0))
    const S = MU.map(() => new Array<number>(K).fill(0))
    for (let i = 0; i < n; i++) {
      const x = Math.floor(r.uniform() * C)
      const u = r.uniform()
      let a = 0
      while (a < K - 1 && u > cum[a]) a++
      N[x][a]++
      if (r.uniform() < MU[x][a]) S[x][a]++
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
  const [size, setSize] = useState<Size>('1000')
  const skew = useParam(3, { min: 0, max: 5, step: 0.1 })
  const misspec = useParam(1, { min: 0, max: 1, step: 0.05 })
  const clip = useParam(10, { min: 1, max: M_MAX, step: 0.5 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const n = Number(size)

  const logging = useMemo(() => softmax(POPULARITY.map((s) => skew.value * s)), [skew.value])
  const weights = useMemo(() => TARGET.map((p) => p.map((q, a) => q / logging[a])), [logging])
  const logs = useMemo(() => simulate(n, logging, seed.value), [n, logging, seed.value])
  const muHat = useMemo(
    () => MU.map((row) => row.map((m, a) => (1 - misspec.value) * m + misspec.value * WRONG[a])),
    [misspec.value],
  )

  const reps = useMemo(
    () => logs.map((log) => estimates(log, n, weights, muHat, clip.value)),
    [logs, n, weights, muHat, clip.value],
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

  const strip: XYSeries[] = [
    {
      name: 'estimates',
      type: 'scatter',
      x: ESTIMATORS.flatMap((_, i) => reps.map((_, j) => i + 1 + 0.6 * (((j * 0.618034) % 1) - 0.5))),
      y: ESTIMATORS.flatMap(({ key }) => reps.map((r) => r[key])),
      group: ESTIMATORS.flatMap((_, i) => reps.map(() => i)),
      groupNames: ESTIMATORS.map((e) => e.label),
    },
    { name: 'true value V(π)', type: 'line', x: [0.5, 5.5], y: [TRUE_VALUE, TRUE_VALUE], emphasis: true, dashed: true },
    {
      name: 'production CTR V(π₀)',
      type: 'line',
      x: [0.5, 5.5],
      y: [productionCtr, productionCtr],
      muted: true,
      dashed: true,
    },
  ]
  const tradeOff: XYSeries[] = [
    { name: 'CIPS RMSE', type: 'line', x: M_GRID, y: clipCurve.map((s) => s.rmse), slot: 1 },
    { name: 'CIPS |bias|', type: 'line', x: M_GRID, y: clipCurve.map((s) => Math.abs(s.bias)), slot: 5 },
    { name: 'CIPS sd', type: 'line', x: M_GRID, y: clipCurve.map((s) => s.sd), slot: 6 },
    { name: 'IPS RMSE', type: 'line', x: [1, M_MAX], y: [stats.ips.rmse, stats.ips.rmse], slot: 0, dashed: true },
    {
      name: 'SNIPS RMSE',
      type: 'line',
      x: [1, M_MAX],
      y: [stats.snips.rmse, stats.snips.rmse],
      slot: 2,
      dashed: true,
    },
  ]

  return (
    <Interactive
      title="Counterfactual estimator lab"
      caption="A production recommender shows one of six items on each of four product pages, with probabilities set by item popularity alone; the propensity skew β sharpens them. A new recommender, which follows the true click rates, is evaluated from 200 simulated logs. Top: the 200 estimates from each estimator against the true click rate of the new recommender (dashed ink) and the production click rate (dashed grey), which is what the log’s own average reports. Bottom: bias, spread and RMSE of clipped IPS as the clip threshold M varies; drag the guide to set M. The reward model used by the direct method and doubly robust blends the true click rates with a model that ignores the page; misspecification 1 is the page-blind model. Raise the skew and IPS spreads out; clip hard and the estimate falls below the truth; misspecify the model and the direct method is confidently wrong while doubly robust stays centred."
      controls={
        <>
          <ParamSlider label="propensity skew β" param={skew} />
          <ParamSlider label="reward-model misspecification" param={misspec} />
          <ParamSlider label="clip threshold M" param={clip} />
          <ParamChoice
            label="impressions per log n"
            value={size}
            onChange={setSize}
            options={SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString() }))}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
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
        <XYChart series={strip} xLabel="estimator" yLabel="estimated click rate" xRange={[0.5, 5.5]} />
        <XYChart
          series={tradeOff}
          xLabel="clip threshold M"
          yLabel="error"
          xRange={[1, M_MAX]}
          yRange={[0, undefined]}
          handles={[{ kind: 'x', at: clip.value, label: 'M', onDrag: (x) => clip.set(x) }]}
        />
      </div>
    </Interactive>
  )
}

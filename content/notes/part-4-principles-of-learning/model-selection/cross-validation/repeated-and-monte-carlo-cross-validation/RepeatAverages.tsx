import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, mean, toFlat } from 'aifn-compute/foundation/tensor'
import { errorRate } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, repeated, shuffleSplit, type Splitter } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { gaussians } from 'aifn-methods/data/synthetic'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { supervised } from '../_shared/problems'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const FITS = 500
const STEP = 10
const CHECKPOINTS = Array.from({ length: FITS / STEP }, (_, i) => (i + 1) * STEP)
const SCHEMES = [
  { value: 'repeated', label: 'repeated 10-fold' },
  { value: 'monte-carlo', label: 'Monte Carlo, 10% held out' },
] as const
type Scheme = (typeof SCHEMES)[number]['value']
const SEPARATION = 2

const splitter = (scheme: Scheme): Splitter =>
  scheme === 'repeated'
    ? repeated(kFold({ k: 10, shuffle: true }), FITS / 10)
    : shuffleSplit({ splits: FITS, testSize: 0.1 })

/**
 * Repeated and Monte Carlo cross-validation of a nearest-neighbour classifier on one dataset: several analysts' running
 * averages as the number of fits grows, the value they all converge to, and the true error of the model fitted to the
 * data.
 */
export function RepeatAverages() {
  const state = useFigureState({
    scheme: choice(SCHEMES, 'repeated', { label: 'scheme' }),
    n: int(60, { ge: 30, le: 300, suggestions: [40, 60, 120, 240], label: 'data points n' }),
    k: int(5, { ge: 1, le: 25, suggestions: [1, 5, 15], label: 'neighbours k' }),
    draws: int(6, { ge: 1, le: 20, suggestions: [3, 6, 12], label: 'analysts (independent runs)' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
  })
  const { scheme, n, k, draws, seed } = state
  const raw = useMemo(() => gaussians(stream(`repeat/${seed}`), { n, separation: SEPARATION }), [n, seed])
  const data = useMemo(() => supervised(raw), [raw])
  const run = useComputed(
    () => {
      const model = kNearestNeighbours({ k })
      const fresh = supervised(gaussians(stream('repeat/fresh'), { n: 4000, separation: SEPARATION }))
      const trueError = errorRate(fresh.y, model.fit(data).decide(fresh.x))
      const lines: number[][] = []
      const all: number[] = []
      for (let a = 0; a < draws; a++) {
        const cv = crossValidate(model, data, splitter(scheme), [errorRate], {
          stream: child(stream('repeat/runs'), a),
        })
        const scores = Array.from(toFlat(cv.scores.errorRate))
        all.push(...scores)
        lines.push(CHECKPOINTS.map((c) => mean(fromData(Float64Array.from(scores.slice(0, c))))))
      }
      const first = lines.map((l) => l[0])
      const last = lines.map((l) => l[l.length - 1])
      return {
        lines,
        trueError,
        bayes: (raw.meta?.truth as { bayesError?: number } | undefined)?.bayesError ?? NaN,
        limit: mean(fromData(Float64Array.from(all))),
        sdFirst: draws > 1 ? standardDeviation(first, { sample: true }) : NaN,
        sdLast: draws > 1 ? standardDeviation(last, { sample: true }) : NaN,
      }
    },
    [data, raw, scheme, k, draws],
    { mode: 'release' },
  )
  const v = run.value
  const fx = useAxis({ label: 'model fits', range: [STEP, FITS], log: true, format: (x) => String(Math.round(x)) })
  const ey = useAxis({ label: 'estimated error rate', hold: 'union', key: `${scheme}/${n}/${k}/${seed}` })
  return (
    <Figure
      title="What repetition averages away"
      state={state}
      caption={`One dataset of n points from two Gaussian classes (Bayes error ${fmt(v.bayes)}, muted dashed), classified by k nearest neighbours. Each thin line is one analyst's estimate of the error rate as the number of model fits grows: 10 fits are one 10-fold cross-validation, 500 are 50 repeats; or, for Monte Carlo cross-validation, one fit per random split holding out 10%. The lines start apart, because one partition gives a noisy estimate, and close in on the same value (dashed, the average over every run), because further partitions of the same data average out the partition noise. That common value is not the true error of the model fitted to these data (ink dashed, measured on 4,000 fresh points): the gap comes from the data, and no amount of repetition removes it. Change the dataset to move the gap.`}
      readouts={
        <>
          <Readout label="spread between analysts after 10 fits (sd)" value={fmt(v.sdFirst)} />
          <Readout label={`spread after ${FITS} fits (sd)`} value={fmt(v.sdLast)} />
          <Readout label="value the runs converge to" value={fmt(v.limit)} />
          <Readout label="true error of the model on these data" value={fmt(v.trueError)} />
        </>
      }
    >
      <Plot x={fx} y={ey} title="running averages of the estimate">
        {v.lines.map((line, a) => (
          <Curve
            key={a}
            name="one analyst's running average"
            x={CHECKPOINTS}
            y={line}
            slot={0}
            thin
            silent
            stale={run.stale}
          />
        ))}
        <Curve name="average over every run" x={[STEP, FITS]} y={[v.limit, v.limit]} slot={1} dashed />
        <Curve
          name="true error of the model on these data"
          x={[STEP, FITS]}
          y={[v.trueError, v.trueError]}
          emphasis
          dashed
        />
        <Curve name="Bayes error" x={[STEP, FITS]} y={[v.bayes, v.bayes]} muted dashed />
      </Plot>
    </Figure>
  )
}

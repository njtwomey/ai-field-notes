import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'

const N = 4
const T = 1000
/** The planted best expert in each quarter of the sequence: it switches three times and returns to expert 1. */
const SCHEDULE = [0, 1, 2, 0]
const SEGMENT = T / SCHEDULE.length
const GOOD = 0.2
const BAD = 0.6
const ROUNDS = Array.from({ length: T }, (_, t) => t + 1)
const EXPERT_SLOTS = [0, 1, 2, 3]
const ALGO_SLOTS = { ftl: 4, hedge: 6, share: 5 }

/** Bernoulli losses: the planted expert of each segment has mean 0.2, the others 0.6. Fixed seed, so only η and α vary. */
function makeLosses(): number[][] {
  const r = stream(7)
  return ROUNDS.map((_, t) => {
    const best = SCHEDULE[Math.floor(t / SEGMENT)]
    return Array.from({ length: N }, (_, i) => (uniform(r) < (i === best ? GOOD : BAD) ? 1 : 0))
  })
}
const LOSSES = makeLosses()

type Run = { regret: number[]; weights: number[][] }

/**
 * Exponential weights with fixed share: after the multiplicative update, a fraction α of the mass is spread uniformly.
 * α = 0 is Hedge. Regret is measured against the planted switching sequence.
 */
function runShare(eta: number, alpha: number): Run {
  let w = new Array<number>(N).fill(1 / N)
  const regret: number[] = []
  const weights: number[][] = []
  let learner = 0
  let comparator = 0
  for (let t = 0; t < T; t++) {
    const l = LOSSES[t]
    weights.push(w)
    learner += w.reduce((a, wi, i) => a + wi * l[i], 0)
    comparator += l[SCHEDULE[Math.floor(t / SEGMENT)]]
    regret.push(learner - comparator)
    const v = w.map((wi, i) => wi * Math.exp(-eta * l[i]))
    const z = v.reduce((a, b) => a + b, 0)
    w = v.map((vi) => alpha / N + (1 - alpha) * (vi / z))
  }
  return { regret, weights }
}

/** Follow the leader: the expert with the smallest cumulative loss so far (lowest index on ties). */
function runFtl(): number[] {
  const cum = new Array<number>(N).fill(0)
  const regret: number[] = []
  let learner = 0
  let comparator = 0
  for (let t = 0; t < T; t++) {
    const leader = cum.indexOf(Math.min(...cum))
    const l = LOSSES[t]
    learner += l[leader]
    comparator += l[SCHEDULE[Math.floor(t / SEGMENT)]]
    regret.push(learner - comparator)
    for (let i = 0; i < N; i++) cum[i] += l[i]
  }
  return regret
}
const FTL_REGRET = runFtl()

/** Fixed-share bound √(T P / 2) with P = (m+1) ln N + m ln(1/α) + (T−m−1) ln(1/(1−α)), at the tuned η. */
const M_SWITCHES = SCHEDULE.length - 1
function fixedShareBound(alpha: number): number {
  const p =
    (M_SWITCHES + 1) * Math.log(N) + M_SWITCHES * Math.log(1 / alpha) + (T - M_SWITCHES - 1) * Math.log(1 / (1 - alpha))
  return Math.sqrt((T * p) / 2)
}

type Shown = 'share' | 'hedge'

/** Cumulative regret against a switching best expert for FTL, Hedge and fixed share, with the weights over time. */
export function TrackingFigure() {
  const state = useFigureState({
    round: float(T, { min: 10, max: T, step: 10, label: 'round t', format: (v) => String(v) }),
    logEta: float(Math.log10(0.3), {
      min: -2,
      max: 1,
      step: 0.05,
      label: 'learning rate η',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    logAlpha: float(Math.log10(M_SWITCHES / (T - 1)), {
      min: -4,
      max: -0.5,
      step: 0.05,
      label: 'share rate α',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    shown: choice<Shown>(
      [
        { value: 'share', label: 'fixed share' },
        { value: 'hedge', label: 'Hedge' },
      ],
      'share',
      { label: 'weights of' },
    ),
  })
  const eta = 10 ** state.logEta
  const alpha = 10 ** state.logAlpha
  const hedge = useMemo(() => runShare(eta, 0), [eta])
  const share = useMemo(() => runShare(eta, alpha), [eta, alpha])
  const t = state.round
  const xs = ROUNDS.slice(0, t)

  const regretSeries = [
    { name: 'follow the leader', x: xs, y: FTL_REGRET.slice(0, t), slot: ALGO_SLOTS.ftl },
    { name: 'Hedge', x: xs, y: hedge.regret.slice(0, t), slot: ALGO_SLOTS.hedge },
    { name: 'fixed share', x: xs, y: share.regret.slice(0, t), slot: ALGO_SLOTS.share },
  ] as const
  const run = state.shown === 'share' ? share : hedge
  const weightSeries: SeriesSpec[] = EXPERT_SLOTS.map((slot, i) => ({
    name: `expert ${i + 1}`,
    type: 'line',
    x: xs,
    y: run.weights.slice(0, t).map((w) => w[i]),
    slot,
  }))

  const xAxis = useAxis({ label: 'round t', range: [0, T] })
  const yAxis = useAxis({ label: 'regret vs switching sequence', hold: 'union' })
  const xAxis2 = useAxis({ label: 'round t', range: [0, T] })
  const yAxis2 = useAxis({ label: 'weight', range: [0, 1] })
  return (
    <Figure
      title="Tracking a best expert that switches"
      state={state}
      caption="Four experts with 0/1 losses. In each quarter of the 1000 rounds one expert errs with probability 0.2 and the others with 0.6; the good expert goes 1, 2, 3, then back to 1. Left: cumulative loss minus the loss of that switching sequence. Hedge's weight on a once-poor expert decays exponentially, so after each switch it needs longer to move; fixed share keeps every weight above α/N and moves after a few dozen rounds. Follow the leader waits until the new expert's total overtakes the old one's. Step through the rounds to watch the weights move; a larger α tracks faster but pays more on every stationary stretch."

      readouts={
        <>
          <Readout label="FTL" value={formatNumber(FTL_REGRET[t - 1])} />
          <Readout label="Hedge" value={formatNumber(hedge.regret[t - 1])} />
          <Readout label="fixed share" value={formatNumber(share.regret[t - 1])} />
          <Readout label="fixed-share bound at T (tuned η)" value={formatNumber(fixedShareBound(alpha))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...regretSeries[0]} />
          <Curve {...regretSeries[1]} />
          <Curve {...regretSeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(weightSeries)}
        </Plot>
      </div>
    </Figure>
  )
}

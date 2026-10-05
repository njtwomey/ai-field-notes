import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { BEST_VALUE, bestPolicy, estimate, logData, type Estimates, type RewardModel } from '../_shared/ope'

const EPS_GRID = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1]
const REPS = 120
const SIZES = ['200', '1000', '5000'] as const
type Size = (typeof SIZES)[number]
type Key = 'ips' | 'snips' | 'dm' | 'dr'
const ESTIMATORS: { key: Key; label: string; slot: number }[] = [
  { key: 'ips', label: 'IPS', slot: 0 },
  { key: 'snips', label: 'SNIPS', slot: 1 },
  { key: 'dm', label: 'direct method', slot: 2 },
  { key: 'dr', label: 'doubly robust', slot: 3 },
]

/** REPS independent logs of size n at exploration rate ε, each reduced to the four estimates. */
function replicate(n: number, eps: number, model: RewardModel, seed: number): Estimates[] {
  return Array.from({ length: REPS }, (_, j) =>
    estimate(logData(n, eps, stream(seed * 7919 + j * 31 + 1)), bestPolicy, model),
  )
}

const rmse = (xs: number[]) => Math.sqrt(xs.reduce((acc, v) => acc + (v - BEST_VALUE) ** 2, 0) / xs.length)
const meanOf = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

/**
 * Sampling distributions of IPS, SNIPS, the direct method and doubly robust estimates of a target policy's value, as
 * the logging policy's overlap with the target shrinks.
 */
export function OpeVariance() {
  const state = useFigureState({
    eps: float(0.1, { min: 0.02, max: 1, step: 0.01, label: 'logging exploration ε' }),
    size: choice<Size>(
      SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString() })),
      '1000',
      { label: 'logged rows n' },
    ),
    model: choice<RewardModel>(
      [
        { value: 'constant', label: 'constant per arm (wrong)' },
        { value: 'linear', label: 'linear in x (right)' },
      ],
      'constant',
      { label: 'reward model (DM, DR)' },
    ),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const n = Number(state.size)

  const curves = useMemo(
    () =>
      EPS_GRID.map((e) => {
        const reps = replicate(n, e, state.model, state.seed)
        return Object.fromEntries(ESTIMATORS.map(({ key }) => [key, rmse(reps.map((r) => r[key]))])) as Record<
          Key,
          number
        >
      }),
    [n, state.model, state.seed],
  )
  const reps = useMemo(
    () => replicate(n, state.eps, state.model, state.seed + 1000),
    [n, state.eps, state.model, state.seed],
  )

  const rmseSeries: SeriesSpec[] = ESTIMATORS.map(({ key, label, slot }) => ({
    name: label,
    type: 'line',
    x: EPS_GRID,
    y: curves.map((c) => c[key]),
    slot,
  }))
  // Strip plot: each replicate's estimate, spread horizontally around its estimator's position.
  const strip: SeriesSpec[] = [
    ...ESTIMATORS.map(({ key, label, slot }, i): SeriesSpec => ({
      name: label,
      type: 'scatter',
      x: reps.map((_, j) => i + 1 + 0.6 * (((j * 0.618034) % 1) - 0.5)),
      y: reps.map((r) => r[key]),
      slot,
    })),
    { name: 'true value V(π)', type: 'line', x: [0.5, 4.5], y: [BEST_VALUE, BEST_VALUE], emphasis: true, dashed: true },
  ]
  const ess = meanOf(reps.map((r) => r.ess))

  const xAxis = useAxis({ label: 'logging exploration ε', range: [0, 1] })
  const yAxis = useAxis({ label: 'RMSE', hold: 'union', log: true })
  const xAxis2 = useAxis({ label: 'estimator', range: [0.5, 4.5] })
  const yAxis2 = useAxis({ label: 'estimate of V(π)', hold: 'union' })
  return (
    <Figure
      title="Estimator error and overlap"
      state={state}
      caption="A target policy (the best policy for three arms whose mean reward is linear in a context x) is evaluated from logs of a policy that plays a poor arm with probability 1 − ε and a uniformly random arm otherwise. With small ε the target's actions are rarely logged, so importance weights reach 3/ε. Left: root-mean-squared error of each estimator over 120 simulated logs, on a log scale; drag the guide to set ε. Right: the 120 estimates at that ε against the true value. IPS is unbiased but its spread explodes as overlap shrinks, and it can exceed 1. SNIPS trades a small bias for much less spread. The direct method with a constant-per-arm model is stable but biased whatever ε; with the correct linear model it is nearly exact. Doubly robust corrects the model's bias with weighted residuals, so it inherits IPS's variance only in proportion to the residuals."

      readouts={
        <>
          <Readout label="true value V(π)" value={formatNumber(BEST_VALUE)} />
          {ESTIMATORS.map(({ key, label }) => {
            const xs = reps.map((r) => r[key])
            const m = meanOf(xs)
            const sd = Math.sqrt(meanOf(xs.map((v) => (v - m) ** 2)))
            return (
              <Readout
                key={key}
                label={`${label}: bias, sd`}
                value={`${formatNumber(m - BEST_VALUE)}, ${formatNumber(sd)}`}
              />
            )
          })}
          <Readout label="effective sample size (mean)" value={`${formatNumber(ess)} of ${n}`} />
          <Readout label="largest weight 3/ε" value={formatNumber(3 / state.eps)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(rmseSeries)}
          <Handle {...state.handle('eps', { label: 'ε' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(strip)}
        </Plot>
      </div>
    </Figure>
  )
}

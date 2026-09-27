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
    estimate(logData(n, eps, rng(seed * 7919 + j * 31 + 1)), bestPolicy, model),
  )
}

const rmse = (xs: number[]) => Math.sqrt(xs.reduce((acc, v) => acc + (v - BEST_VALUE) ** 2, 0) / xs.length)
const meanOf = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

/**
 * Sampling distributions of IPS, SNIPS, the direct method and doubly robust estimates of a target policy's value, as
 * the logging policy's overlap with the target shrinks.
 */
export function OpeVariance() {
  const [size, setSize] = useState<Size>('1000')
  const [model, setModel] = useState<RewardModel>('constant')
  const eps = useParam(0.1, { min: 0.02, max: 1, step: 0.01 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const n = Number(size)

  const curves = useMemo(
    () =>
      EPS_GRID.map((e) => {
        const reps = replicate(n, e, model, seed.value)
        return Object.fromEntries(ESTIMATORS.map(({ key }) => [key, rmse(reps.map((r) => r[key]))])) as Record<
          Key,
          number
        >
      }),
    [n, model, seed.value],
  )
  const reps = useMemo(() => replicate(n, eps.value, model, seed.value + 1000), [n, eps.value, model, seed.value])

  const rmseSeries: XYSeries[] = ESTIMATORS.map(({ key, label, slot }) => ({
    name: label,
    type: 'line',
    x: EPS_GRID,
    y: curves.map((c) => c[key]),
    slot,
  }))
  // Strip plot: each replicate's estimate, spread horizontally around its estimator's position.
  const strip: XYSeries[] = [
    ...ESTIMATORS.map(({ key, label, slot }, i): XYSeries => ({
      name: label,
      type: 'scatter',
      x: reps.map((_, j) => i + 1 + 0.6 * (((j * 0.618034) % 1) - 0.5)),
      y: reps.map((r) => r[key]),
      slot,
    })),
    { name: 'true value V(π)', type: 'line', x: [0.5, 4.5], y: [BEST_VALUE, BEST_VALUE], emphasis: true, dashed: true },
  ]
  const ess = meanOf(reps.map((r) => r.ess))

  return (
    <Interactive
      title="Estimator error and overlap"
      caption="A target policy (the best policy for three arms whose mean reward is linear in a context x) is evaluated from logs of a policy that plays a poor arm with probability 1 − ε and a uniformly random arm otherwise. With small ε the target's actions are rarely logged, so importance weights reach 3/ε. Left: root-mean-squared error of each estimator over 120 simulated logs, on a log scale; drag the guide to set ε. Right: the 120 estimates at that ε against the true value. IPS is unbiased but its spread explodes as overlap shrinks, and it can exceed 1. SNIPS trades a small bias for much less spread. The direct method with a constant-per-arm model is stable but biased whatever ε; with the correct linear model it is nearly exact. Doubly robust corrects the model's bias with weighted residuals, so it inherits IPS's variance only in proportion to the residuals."
      controls={
        <>
          <ParamSlider label="logging exploration ε" param={eps} />
          <ParamChoice
            label="logged rows n"
            value={size}
            onChange={setSize}
            options={SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString() }))}
          />
          <ParamChoice
            label="reward model (DM, DR)"
            value={model}
            onChange={setModel}
            options={[
              { value: 'constant', label: 'constant per arm (wrong)' },
              { value: 'linear', label: 'linear in x (right)' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
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
          <Readout label="largest weight 3/ε" value={formatNumber(3 / eps.value)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={rmseSeries}
          xLabel="logging exploration ε"
          yLabel="RMSE"
          xRange={[0, 1]}
          yLog
          handles={[{ kind: 'x', at: eps.value, label: 'ε', onDrag: (x) => eps.set(x) }]}
        />
        <XYChart series={strip} xLabel="estimator" yLabel="estimate of V(π)" xRange={[0.5, 4.5]} />
      </div>
    </Interactive>
  )
}

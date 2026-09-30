import { Normal } from 'aifn/distributions'
import {
  clutterEp,
  clutterLogLikelihood,
  clutterPosterior,
  epLogEvidence,
  expectationPropagationSteps,
  sampleClutter,
} from 'aifn/ep'
import { stream } from 'aifn/random'
import { linspace, toFlat } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { Player, Slider, Switch, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, formatNumber, type XYSeries } from '@lab/viz'

const GRID = linspace(-4, 8, 481)
const THETA = toFlat(GRID)
const density = (mean: number, variance: number) =>
  Number.isFinite(mean) && variance > 0 ? toFlat(Normal(mean, Math.sqrt(variance)).prob(GRID)) : THETA.map(() => NaN)

/** EP on Minka's clutter problem, one site update per step: the cavity, the tilted distribution and the new q. */
export function ClutterSpecimen() {
  const n = useParam(15, { min: 3, max: 40, step: 1 })
  const w = useParam(0.3, { min: 0.05, max: 0.8, step: 0.05 })
  const damping = useParam(0, { min: 0, max: 0.9, step: 0.05 })
  const seed = useParam(3, { min: 1, max: 30, step: 1 })
  const [showExact, setShowExact] = useState(true)
  const problem = useMemo(() => ({ weight: w.value }), [w.value])
  const x = useMemo(
    () => Array.from(sampleClutter(stream('clutter').child(seed.value), n.value, 2, problem).data),
    [n.value, seed.value, problem],
  )
  const run = useMemo(
    () => trace(expectationPropagationSteps, clutterEp(x, problem, { damping: damping.value }), 30 * x.length),
    [x, problem, damping.value],
  )
  const exact = useMemo(() => clutterPosterior(x, problem, { lower: -4, upper: 8, points: 481 }), [x, problem])
  const [step, setStep] = useState(18)
  const s = run.steps[Math.min(step, run.steps.length - 1)]
  const q = s.posterior
  const i = s.site
  const factor = i >= 0 ? THETA.map((t) => Math.exp(clutterLogLikelihood(t, [x[i]], problem))) : []
  const qd = density(q.mean, q.variance)
  const top = Math.max(...qd)
  const fmax = Math.max(...factor, 1e-300)
  const series: XYSeries[] = [
    ...(showExact
      ? [{ name: 'exact posterior', type: 'area' as const, x: THETA, y: toFlat(exact.density), muted: true }]
      : []),
    ...(i >= 0
      ? [
          {
            name: `factor t_${i}(θ), scaled`,
            type: 'line' as const,
            x: THETA,
            y: factor.map((f) => (f / fmax) * top),
            slot: 3,
            dashed: true,
          },
          {
            name: 'cavity q∖i',
            type: 'line' as const,
            x: THETA,
            y: density(s.cavity.mean, s.cavity.variance),
            slot: 2,
          },
          {
            name: 'tilted, moment-matched',
            type: 'line' as const,
            x: THETA,
            y: density(s.tiltedMoments.mean, s.tiltedMoments.variance),
            slot: 1,
            dashed: true,
          },
        ]
      : []),
    { name: 'EP posterior q', type: 'line', x: THETA, y: qd, slot: 0 },
    { name: 'data', type: 'scatter', x, y: x.map(() => 0), emphasis: true },
  ]
  const format = (p: number) => {
    const k = run.index[p]
    return k === 0 ? 'start' : `s${Math.floor((k - 1) / x.length) + 1} i${(k - 1) % x.length}`
  }
  return (
    <Figure
      title="EP on the clutter problem, site by site"
      description="Each update divides a site out of q, multiplies the exact factor back in, and projects the tilted result onto a Gaussian."
      defaultSize="L"
      controls={
        <>
          <Slider label="points n" param={n} />
          <Slider label="clutter probability w" param={w} />
          <Slider label="seed" param={seed} />
          <Slider label="damping" param={damping} />
          <Switch label="show the exact posterior" checked={showExact} onChange={setShowExact} />
          <div className="col-span-full">
            <Player
              label="update (sweep s, site i)"
              value={step}
              onChange={setStep}
              count={run.steps.length}
              format={format}
              defaultSpeed={4}
            />
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="x_i" value={i >= 0 ? formatNumber(x[i]) : '—'} />
          <Readout label="EP mean / sd" value={`${formatNumber(q.mean)} / ${formatNumber(Math.sqrt(q.variance))}`} />
          <Readout
            label="exact mean / sd"
            value={`${formatNumber(exact.mean)} / ${formatNumber(Math.sqrt(exact.variance))}`}
          />
          <Readout
            label="last sweep's largest change"
            value={Number.isFinite(s.lastSweepChange) ? formatNumber(s.lastSweepChange) : '—'}
          />
          <Readout
            label="log evidence EP / exact"
            value={`${formatNumber(epLogEvidence(s))} / ${formatNumber(exact.logEvidence)}`}
          />
          <Readout label="skipped updates" value={s.totalSkipped} />
        </>
      }
      caption="θ ~ N(0, 100); each point is N(θ, 1) with probability 1 − w and clutter N(0, 10) with probability w (Minka 2001). The data are the ink dots, drawn at θ = 2. At each update the cavity is q with site i removed, the dashed factor is the exact clutter likelihood of x_i, and the tilted distribution (cavity × factor) is replaced by the Gaussian with its mean and variance. The first sweep is assumed density filtering; later sweeps revisit each site until nothing moves. The evidence estimate uses every site's normaliser, so it is meaningful only after the first sweep."
    >
      <XYChart
        series={series}
        xLabel="θ"
        yLabel="density"
        rescaleOnChange={false}
        axisKey={`${n.value}-${seed.value}-${w.value}`}
        holdFit="union"
      />
    </Figure>
  )
}

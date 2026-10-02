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
} from 'aifn-render'
import {
  constantVelocity,
  eigenvalues2,
  ellipse,
  kalmanFilter,
  positionBlock,
  positionRmse,
  simulate,
  steadyStateGain,
  type NoiseShape,
} from '../_shared/linear-gaussian'

const STEPS = 40
/** The noise levels that generated the data. The sliders set what the filter assumes. */
const TRUE_Q = 0.01
const TRUE_R = 1
const M0 = [0, 0, 0, 0]
const P0 = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
]

export function KalmanTracking() {
  const logQ = useParam(Math.log10(TRUE_Q), { min: -3.5, max: 1, step: 0.1 })
  const logR = useParam(0, { min: -1.5, max: 1.5, step: 0.1 })
  const seed = useParam(7, { min: 1, max: 20, step: 1 })
  const step = useParam(20, { min: 1, max: STEPS, step: 1 })
  const [shape, setShape] = useState<NoiseShape>('round')
  const [level, setLevel] = useState<'1' | '2'>('2')
  const k = Number(level)

  const data = useMemo(() => {
    const world = simulate(constantVelocity(TRUE_Q, TRUE_R, shape), STEPS, seed.value)
    const px = world.z.map((s) => s[0])
    const py = world.z.map((s) => s[1])
    // A square window around the true path, so the equal-aspect plot keeps a fixed frame as t changes.
    // Rounded to multiples of 5 so the axis ends fall on tick marks.
    const cx = 5 * Math.round((Math.min(...px) + Math.max(...px)) / 10)
    const cy = 5 * Math.round((Math.min(...py) + Math.max(...py)) / 10)
    const half =
      5 * Math.ceil((Math.max(Math.max(...px) - Math.min(...px), Math.max(...py) - Math.min(...py)) / 2 + 5) / 5)
    return { ...world, cx, cy, half }
  }, [shape, seed.value])

  const fit = useMemo(() => {
    const model = constantVelocity(10 ** logQ.value, 10 ** logR.value, shape)
    const steps = kalmanFilter(model, data.x, M0, P0)
    const gain = eigenvalues2(positionBlock(steadyStateGain(model)))
    return {
      model,
      steps,
      gain,
      filterRmse: positionRmse(
        steps.map((s) => s.m),
        data.z,
      ),
      measurementRmse: positionRmse(data.x, data.z),
    }
  }, [data, logQ.value, logR.value, shape])

  const t = step.value - 1
  const upTo = <T,>(xs: T[]) => xs.slice(0, t + 1)
  const now = fit.steps[t]
  const contour = (name: string, mean: number[], cov: number[][], slot: number, dashed = false): XYSeries => ({
    name,
    type: 'line',
    ...ellipse(mean, cov, k),
    slot,
    dashed,
  })
  const series: XYSeries[] = [
    {
      name: 'true path',
      type: 'line',
      x: upTo(data.z).map((s) => s[0]),
      y: upTo(data.z).map((s) => s[1]),
      emphasis: true,
    },
    {
      name: 'measurements',
      type: 'scatter',
      x: upTo(data.x).map((v) => v[0]),
      y: upTo(data.x).map((v) => v[1]),
      slot: 2,
    },
    {
      name: 'filtered estimate',
      type: 'line',
      x: upTo(fit.steps).map((s) => s.m[0]),
      y: upTo(fit.steps).map((s) => s.m[1]),
      slot: 0,
    },
    // Earlier posterior contours every fifth step, to show the uncertainty settling to its steady state.
    ...fit.steps
      .map((s, i) => ({ s, i }))
      .filter(({ i }) => i < t && i % 5 === 4)
      .map(({ s }) => contour('posterior', s.m, positionBlock(s.P), 0)),
    contour('prediction at t', now.mPred, positionBlock(now.PPred), 1, true),
    contour('measurement at t', data.x[t], fit.model.R, 2, true),
    contour('posterior', now.m, positionBlock(now.P), 0),
  ]
  const gainSeries: XYSeries[] = [
    {
      name: 'position gain (largest eigenvalue)',
      type: 'line',
      x: fit.steps.map((_, i) => i + 1),
      y: fit.steps.map((s) => eigenvalues2(positionBlock(s.K))[0]),
      slot: 0,
    },
    { name: 'step t', type: 'scatter', x: [t + 1], y: [eigenvalues2(positionBlock(now.K))[0]], emphasis: true },
  ]
  const gainText =
    Math.abs(fit.gain[0] - fit.gain[1]) < 1e-3
      ? formatNumber(fit.gain[0])
      : `${formatNumber(fit.gain[1])} to ${formatNumber(fit.gain[0])}`

  return (
    <Interactive
      title="Tracking a moving object"
      caption={`An object moves in the plane with randomly changing velocity (true q = ${TRUE_Q}, r = ${TRUE_R}). A sensor reports its position with noise (dots). At step t the filter predicts where the object is (the dashed contour around the prediction), weighs that prediction against the new measurement and its noise (the dashed contour around the measurement), and keeps the posterior (solid). The posterior is tighter than both. The sliders set the noise the filter assumes. A small q trusts the motion model, so the estimate is smooth but lags turns. A large q trusts the measurements and follows their noise. The filter error is lowest near the true values.`}
      controls={
        <>
          <ParamSlider label="step t" param={step} format={(v) => String(v)} withArrows />
          <ParamSlider label="assumed process noise q" param={logQ} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="assumed measurement noise r" param={logR} format={(v) => formatNumber(10 ** v)} />
          <ParamChoice
            label="measurement noise"
            value={shape}
            onChange={setShape}
            options={[
              { value: 'round', label: 'round' },
              { value: 'elongated', label: 'elongated' },
            ]}
          />
          <ParamChoice
            label="contours"
            value={level}
            onChange={setLevel}
            options={[
              { value: '1', label: '1σ' },
              { value: '2', label: '2σ' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="RMSE, raw measurements" value={formatNumber(fit.measurementRmse)} />
          <Readout label="RMSE, filter" value={formatNumber(fit.filterRmse)} />
          <Readout label="steady-state position gain" value={gainText} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="mx-auto w-full max-w-xl">
          <XYChart
            series={series}
            xLabel="x"
            yLabel="y"
            xRange={[data.cx - data.half, data.cx + data.half]}
            yRange={[data.cy - data.half, data.cy + data.half]}
            equalAspect
          />
        </div>
        <XYChart series={gainSeries} xLabel="step t" yLabel="gain" yRange={[0, 1]} height={160} />
      </div>
    </Interactive>
  )
}

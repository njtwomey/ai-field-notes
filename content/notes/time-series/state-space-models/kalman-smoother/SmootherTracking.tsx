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
import {
  constantVelocity,
  ellipse,
  kalmanFilter,
  positionBlock,
  positionRmse,
  rtsSmoother,
  simulate,
  type Vec,
  type Mat,
} from '../_shared/linear-gaussian'

const STEPS = 40
const Q = 0.01
const M0 = [0, 0, 0, 0]
const P0 = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
]

type Show = 'both' | 'filter' | 'smoother'

export function SmootherTracking() {
  const logR = useParam(0, { min: -1, max: 1, step: 0.1 })
  const seed = useParam(7, { min: 1, max: 20, step: 1 })
  const [show, setShow] = useState<Show>('both')

  const r = useMemo(() => {
    const model = constantVelocity(Q, 10 ** logR.value, 'round')
    const world = simulate(model, STEPS, seed.value)
    const filtered = kalmanFilter(model, world.x, M0, P0)
    const smoothed = rtsSmoother(model, filtered)
    const px = world.z.map((s) => s[0])
    const py = world.z.map((s) => s[1])
    // Rounded to multiples of 5 so the axis ends fall on tick marks.
    const cx = 5 * Math.round((Math.min(...px) + Math.max(...px)) / 10)
    const cy = 5 * Math.round((Math.min(...py) + Math.max(...py)) / 10)
    const half =
      5 * Math.ceil((Math.max(Math.max(...px) - Math.min(...px), Math.max(...py) - Math.min(...py)) / 2 + 5) / 5)
    return {
      ...world,
      filtered,
      smoothed,
      cx,
      cy,
      half,
      rmse: {
        measurements: positionRmse(world.x, world.z),
        filter: positionRmse(
          filtered.map((s) => s.m),
          world.z,
        ),
        smoother: positionRmse(
          smoothed.map((s) => s.m),
          world.z,
        ),
      },
    }
  }, [logR.value, seed.value])

  const path = (name: string, estimates: { m: Vec; P: Mat }[], slot: number): XYSeries[] => [
    { name, type: 'line', x: estimates.map((s) => s.m[0]), y: estimates.map((s) => s.m[1]), slot },
    // 2σ contours every fifth step.
    ...estimates
      .filter((_, i) => i % 5 === 2)
      .map((s): XYSeries => ({ name, type: 'line', ...ellipse(s.m, positionBlock(s.P), 2), slot, dashed: true })),
  ]
  const plane: XYSeries[] = [
    { name: 'true path', type: 'line', x: r.z.map((s) => s[0]), y: r.z.map((s) => s[1]), emphasis: true },
    { name: 'measurements', type: 'scatter', x: r.x.map((v) => v[0]), y: r.x.map((v) => v[1]), slot: 2 },
    ...(show !== 'smoother' ? path('filter', r.filtered, 0) : []),
    ...(show !== 'filter' ? path('smoother', r.smoothed, 1) : []),
  ]
  const t = r.z.map((_, i) => i + 1)
  const spread: XYSeries[] = [
    { name: 'filter', type: 'line', x: t, y: r.filtered.map((s) => Math.sqrt(s.P[0][0])), slot: 0 },
    { name: 'smoother', type: 'line', x: t, y: r.smoothed.map((s) => Math.sqrt(s.P[0][0])), slot: 1 },
  ]

  return (
    <Interactive
      title="Filtering and smoothing the same track"
      caption="The same constant-velocity track as in the Kalman filter note, with the filter's noise settings equal to the true ones. The filter estimates each position from the measurements up to that time. The smoother uses all of them. Dashed contours are 2σ regions every fifth step. The lower chart gives the posterior standard deviation of the x-position: the smoother's is smaller everywhere except at the last step, where the two coincide, and it is largest near the ends, which have data on one side only."
      controls={
        <>
          <ParamSlider label="measurement noise r" param={logR} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="show"
            value={show}
            onChange={setShow}
            options={[
              { value: 'both', label: 'both' },
              { value: 'filter', label: 'filter' },
              { value: 'smoother', label: 'smoother' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="RMSE, raw measurements" value={formatNumber(r.rmse.measurements)} />
          <Readout label="RMSE, filter" value={formatNumber(r.rmse.filter)} />
          <Readout label="RMSE, smoother" value={formatNumber(r.rmse.smoother)} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="mx-auto w-full max-w-xl">
          <XYChart
            series={plane}
            xLabel="x"
            yLabel="y"
            xRange={[r.cx - r.half, r.cx + r.half]}
            yRange={[r.cy - r.half, r.cy + r.half]}
            equalAspect
          />
        </div>
        <XYChart series={spread} xLabel="step t" yLabel="sd of x-position" yRange={[0, undefined]} height={180} />
      </div>
    </Interactive>
  )
}

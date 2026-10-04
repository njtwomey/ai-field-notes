import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
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
  const state = useFigureState({
    logR: float(0, {
      min: -1,
      max: 1,
      step: 0.1,
      label: 'measurement noise r',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    seed: int(7, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
    show: choice<Show>(
      [
        { value: 'both', label: 'both' },
        { value: 'filter', label: 'filter' },
        { value: 'smoother', label: 'smoother' },
      ],
      'both',
      { label: 'show' },
    ),
  })

  const r = useMemo(() => {
    const model = constantVelocity(Q, 10 ** state.logR, 'round')
    const world = simulate(model, STEPS, state.seed)
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
  }, [state.logR, state.seed])

  const path = (name: string, estimates: { m: Vec; P: Mat }[], slot: number): SeriesSpec[] => [
    { name, type: 'line', x: estimates.map((s) => s.m[0]), y: estimates.map((s) => s.m[1]), slot },
    // 2σ contours every fifth step.
    ...estimates
      .filter((_, i) => i % 5 === 2)
      .map((s): SeriesSpec => ({ name, type: 'line', ...ellipse(s.m, positionBlock(s.P), 2), slot, dashed: true })),
  ]
  const plane: SeriesSpec[] = [
    { name: 'true path', type: 'line', x: r.z.map((s) => s[0]), y: r.z.map((s) => s[1]), emphasis: true },
    { name: 'measurements', type: 'scatter', x: r.x.map((v) => v[0]), y: r.x.map((v) => v[1]), slot: 2 },
    ...(state.show !== 'smoother' ? path('filter', r.filtered, 0) : []),
    ...(state.show !== 'filter' ? path('smoother', r.smoothed, 1) : []),
  ]
  const t = r.z.map((_, i) => i + 1)
  const spread = [
    { name: 'filter', x: t, y: r.filtered.map((s) => Math.sqrt(s.P[0][0])), slot: 0 },
    { name: 'smoother', x: t, y: r.smoothed.map((s) => Math.sqrt(s.P[0][0])), slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [r.cx - r.half, r.cx + r.half] })
  const yAxis = useAxis({ label: 'y', range: [r.cy - r.half, r.cy + r.half], equal: xAxis })
  const xAxis2 = useAxis({ label: 'step t', hold: 'union' })
  const yAxis2 = useAxis({ label: 'sd of x-position', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Filtering and smoothing the same track"
      state={state}
      caption="The same constant-velocity track as in the Kalman filter note, with the filter's noise settings equal to the true ones. The filter estimates each position from the measurements up to that time. The smoother uses all of them. Dashed contours are 2σ regions every fifth step. The lower chart gives the posterior standard deviation of the x-position: the smoother's is smaller everywhere except at the last step, where the two coincide, and it is largest near the ends, which have data on one side only."

      readouts={
        <>
          <Readout label="RMSE, raw measurements" value={formatNumber(r.rmse.measurements)} />
          <Readout label="RMSE, filter" value={formatNumber(r.rmse.filter)} />
          <Readout label="RMSE, smoother" value={formatNumber(r.rmse.smoother)} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="mx-auto w-full max-w-xl">
          <Plot x={xAxis} y={yAxis}>
            {seriesLayers(plane)}
          </Plot>
        </div>
        <Plot x={xAxis2} y={yAxis2} height={180}>
          <Curve {...spread[0]} />
          <Curve {...spread[1]} />
        </Plot>
      </div>
    </Figure>
  )
}

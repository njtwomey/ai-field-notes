import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid } from '../_shared/ep'
import { ellipse, epProbit, exactProbitGrid, laplaceProbit, predictive, type V2 } from '../_shared/probit'

const PRIOR_VAR = 4
const SWEEPS = 6
const INITIAL: V2[] = [
  [1, 0.5],
  [2, 1.5],
  [0.5, 2],
  [-1, -0.5],
  [-1.5, 0.5],
  [-0.5, -2],
  [0.8, -0.3],
]
const LABELS = [1, 1, 1, -1, -1, -1, -1]
const W1 = grid(-3, 5, 49)
const W2 = grid(-3, 6, 55)
const XG = grid(-3, 3, 41)
const CHOICES = INITIAL.map((_, i) => ({ value: String(i), label: `${i + 1}` }))

/**
 * EP for 2-D probit regression, site by site. Left: the weight posterior (exact density, with the 2-sd ellipses of EP
 * and Laplace). Right: the data with EP's predictive probability and both decision boundaries.
 */
export function ProbitEp() {
  const [points, setPoints] = useState<V2[]>(INITIAL)
  const [labels, setLabels] = useState(LABELS)
  const state = useFigureState({
    active: choice(CHOICES, '6', { label: 'point to move' }),
  })
  // Site updates made: the walk-through position. It counts updates, so it keeps its meaning when the data change.
  const [step, setStep] = useState(0)

  const r = useMemo(() => {
    const logp = exactProbitGrid(points, labels, PRIOR_VAR, W1, W2)
    const top = Math.max(...logp.flat())
    return {
      exact: logp.map((row) => row.map((v) => Math.exp(v - top))),
      steps: epProbit(points, labels, PRIOR_VAR, SWEEPS),
      laplace: laplaceProbit(points, labels, PRIOR_VAR),
    }
  }, [points, labels])

  const current =
    step === 0 ? { mean: [0, 0] as V2, cov: [PRIOR_VAR, 0, PRIOR_VAR] as [number, number, number] } : r.steps[step - 1]
  const epEllipse = ellipse(current.mean, current.cov)
  const laEllipse = ellipse(r.laplace.mean, r.laplace.cov)
  const weightOverlay = [
    { name: 'EP (2 sd)', x: epEllipse.x, y: epEllipse.y, slot: 2 },
    { name: 'Laplace (2 sd)', x: laEllipse.x, y: laEllipse.y, slot: 3 },
    { name: 'EP mean', x: [current.mean[0]], y: [current.mean[1]], slot: 2 },
    { name: 'Laplace mean', x: [r.laplace.mean[0]], y: [r.laplace.mean[1]], slot: 3 },
  ] as const

  const prob = XG.map((b) => XG.map((a) => predictive(current.mean, current.cov, [a, b])))
  // Decision boundaries wᵀx = 0 through the origin, drawn across the plot.
  const boundary = (m: V2) => {
    const n = Math.hypot(m[0], m[1]) || 1
    const d: V2 = [-m[1] / n, m[0] / n]
    return { x: [-5 * d[0], 5 * d[0]], y: [-5 * d[1], 5 * d[1]] }
  }
  const idx = Number(state.active)
  const others = points.map((p, i) => ({ p, i })).filter(({ i }) => i !== idx)
  const dataOverlay: SeriesSpec[] = [
    { name: 'EP boundary', type: 'line', ...boundary(current.mean), slot: 2 },
    { name: 'Laplace boundary', type: 'line', ...boundary(r.laplace.mean), slot: 3 },
    {
      name: 'data',
      type: 'scatter',
      x: others.map(({ p }) => p[0]),
      y: others.map(({ p }) => p[1]),
      group: others.map(({ i }) => (labels[i] > 0 ? 1 : 0)),
      groupNames: ['y = −1', 'y = +1'],
    },
  ]
  const s = step === 0 ? null : r.steps[step - 1]
  const where = s ? `sweep ${s.sweep + 1}, site ${s.site + 1}${s.ok ? '' : ' (skipped)'}` : 'prior'

  const xAxis = useAxis({ label: 'w₁' })
  const yAxis = useAxis({ label: 'w₂' })
  const xAxis2 = useAxis({ label: 'x₁' })
  const yAxis2 = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="EP for probit regression, site by site"
      state={state}
      caption="Left: the exact posterior density of the weights (shaded), with the 2-sd ellipses and means of EP after the chosen number of site updates and of the Laplace approximation. Right: the data, EP's predictive probability of y = +1 (shaded, 0.5 in the middle of the scale) and both decision boundaries wᵀx = 0. Choose a point, then drag it on the right-hand chart (the ink marker) or flip its label."
      controls={
        <>
          <Player
            value={step}
            onChange={setStep}
            count={INITIAL.length * SWEEPS + 1}
            label="site updates"
            format={(k) => `${k} of ${INITIAL.length * SWEEPS}`}
          />
          <Button variant="outline" size="sm" onClick={() => setLabels((ls) => ls.map((l, j) => (j === idx ? -l : l)))}>
            Flip label of point {idx + 1} (now {labels[idx] > 0 ? '+1' : '−1'})
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setPoints(INITIAL)
              setLabels(LABELS)
            }}
          >
            Reset data
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="position" value={where} />
          <Readout label="EP mean" value={`(${formatNumber(current.mean[0])}, ${formatNumber(current.mean[1])})`} />
          <Readout
            label="Laplace mean"
            value={`(${formatNumber(r.laplace.mean[0])}, ${formatNumber(r.laplace.mean[1])})`}
          />
          <Readout
            label={`p(y = +1) at point ${idx + 1}: EP, Laplace`}
            value={`${formatNumber(predictive(current.mean, current.cov, points[idx]))}, ${formatNumber(predictive(r.laplace.mean, r.laplace.cov, points[idx]))}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={W1} y={W2} z={r.exact} valueLabel={'exact density (relative)'} />
          <Curve {...weightOverlay[0]} live />
          <Curve {...weightOverlay[1]} live />
          <Points {...weightOverlay[2]} live />
          <Points {...weightOverlay[3]} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Raster x={XG} y={XG} z={prob} scale={'diverging'} range={[0, 1]} valueLabel={'p(y = +1)'} />
          {seriesLayers(dataOverlay, { live: true })}
          <Handle
            kind="point"
            at={points[idx]}
            label={`point ${idx + 1}`}
            onDrag={([a, b]) =>
              setPoints((ps) =>
                ps.map((p, j) =>
                  j === idx
                    ? [
                        Math.round(Math.min(2.9, Math.max(-2.9, a)) * 20) / 20,
                        Math.round(Math.min(2.9, Math.max(-2.9, b)) * 20) / 20,
                      ]
                    : p,
                ),
              )
            }
          />
        </Plot>
      </div>
    </Figure>
  )
}

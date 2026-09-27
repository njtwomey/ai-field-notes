import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from '@/components/viz'
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
  const [active, setActive] = useState('6')
  const step = useParam(INITIAL.length * 3, { min: 0, max: INITIAL.length * SWEEPS, step: 1 })

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
    step.value === 0
      ? { mean: [0, 0] as V2, cov: [PRIOR_VAR, 0, PRIOR_VAR] as [number, number, number] }
      : r.steps[step.value - 1]
  const epEllipse = ellipse(current.mean, current.cov)
  const laEllipse = ellipse(r.laplace.mean, r.laplace.cov)
  const weightOverlay: HeatmapOverlay[] = [
    { name: 'EP (2 sd)', type: 'line', x: epEllipse.x, y: epEllipse.y, slot: 2 },
    { name: 'Laplace (2 sd)', type: 'line', x: laEllipse.x, y: laEllipse.y, slot: 3 },
    { name: 'EP mean', type: 'scatter', x: [current.mean[0]], y: [current.mean[1]], slot: 2 },
    { name: 'Laplace mean', type: 'scatter', x: [r.laplace.mean[0]], y: [r.laplace.mean[1]], slot: 3 },
  ]

  const prob = XG.map((b) => XG.map((a) => predictive(current.mean, current.cov, [a, b])))
  // Decision boundaries wᵀx = 0 through the origin, drawn across the plot.
  const boundary = (m: V2) => {
    const n = Math.hypot(m[0], m[1]) || 1
    const d: V2 = [-m[1] / n, m[0] / n]
    return { x: [-5 * d[0], 5 * d[0]], y: [-5 * d[1], 5 * d[1]] }
  }
  const idx = Number(active)
  const others = points.map((p, i) => ({ p, i })).filter(({ i }) => i !== idx)
  const dataOverlay: HeatmapOverlay[] = [
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
  const handles: Handle[] = [
    {
      kind: 'point',
      at: points[idx],
      label: `point ${idx + 1}`,
      onDrag: ([a, b]) =>
        setPoints((ps) =>
          ps.map((p, j) =>
            j === idx
              ? [
                  Math.round(Math.min(2.9, Math.max(-2.9, a)) * 20) / 20,
                  Math.round(Math.min(2.9, Math.max(-2.9, b)) * 20) / 20,
                ]
              : p,
          ),
        ),
    },
  ]
  const s = step.value === 0 ? null : r.steps[step.value - 1]
  const where = s ? `sweep ${s.sweep + 1}, site ${s.site + 1}${s.ok ? '' : ' (skipped)'}` : 'prior'

  return (
    <Interactive
      title="EP for probit regression, site by site"
      caption="Left: the exact posterior density of the weights (shaded), with the 2-sd ellipses and means of EP after the chosen number of site updates and of the Laplace approximation. Right: the data, EP's predictive probability of y = +1 (shaded, 0.5 in the middle of the scale) and both decision boundaries wᵀx = 0. Choose a point, then drag it on the right-hand chart (the ink marker) or flip its label."
      controls={
        <>
          <ParamSlider label="site updates" param={step} withArrows format={(v) => String(v)} />
          <ParamChoice label="point to move" value={active} onChange={setActive} options={CHOICES} />
          <ParamSwitch
            label={`point ${idx + 1} has label +1`}
            checked={labels[idx] > 0}
            onChange={(c) => setLabels((ls) => ls.map((l, j) => (j === idx ? (c ? 1 : -1) : l)))}
          />
          <ParamButton
            onClick={() => {
              setPoints(INITIAL)
              setLabels(LABELS)
            }}
          >
            Reset data
          </ParamButton>
        </>
      }
      readout={
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
        <Heatmap
          x={W1}
          y={W2}
          z={r.exact}
          xLabel="w₁"
          yLabel="w₂"
          overlay={weightOverlay}
          valueLabel="exact density (relative)"
          height={340}
        />
        <Heatmap
          x={XG}
          y={XG}
          z={prob}
          xLabel="x₁"
          yLabel="x₂"
          scale="diverging"
          range={[0, 1]}
          overlay={dataOverlay}
          handles={handles}
          valueLabel="p(y = +1)"
          height={340}
        />
      </div>
    </Interactive>
  )
}

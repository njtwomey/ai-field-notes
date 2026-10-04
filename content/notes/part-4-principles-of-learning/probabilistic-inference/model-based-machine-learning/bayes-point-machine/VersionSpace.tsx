import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid } from '../_shared/gaussian'
import { epBpm, maxMarginDirection, posteriorGrid, type Labelled, type Vec2 } from '../_shared/bpm'

const START: Labelled[] = [
  { x: [1.0, 2.0], y: 1 },
  { x: [2.0, 0.8], y: 1 },
  { x: [0.4, 2.6], y: 1 },
  { x: [-1.5, -0.5], y: -1 },
  { x: [-0.3, -1.8], y: -1 },
  { x: [1.6, -1.2], y: -1 },
]
const WS = grid(-3, 3, 61)
const angle = (a: Vec2, b: Vec2) =>
  (Math.acos(
    Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1])))),
  ) *
    180) /
  Math.PI
/** A boundary through the origin perpendicular to w, clipped to the plot. */
const boundary = (w: Vec2, r = 3): { x: number[]; y: number[] } => {
  const n = Math.hypot(w[0], w[1]) || 1
  const d: Vec2 = [-w[1] / n, w[0] / n]
  return { x: [-r * d[0] * 1.5, r * d[0] * 1.5], y: [-r * d[1] * 1.5, r * d[1] * 1.5] }
}

/** Data space and weight space side by side: the Bayes point against the maximum-margin direction. */
export function VersionSpace() {
  const [data, setData] = useState(START)
  const state = useFigureState({
    beta: float(0.2, { min: 0.05, max: 1.5, step: 0.05, label: 'label noise β', format: (v) => v.toFixed(2) }),
  })

  const { ep, exact, svm, z } = useMemo(() => {
    const g = posteriorGrid(data, state.beta, WS)
    return { ep: epBpm(data, state.beta, 30), exact: g.mean, svm: maxMarginDirection(data), z: g.z }
  }, [data, state.beta])

  const dataSeries = useMemo(() => {
    const bp = boundary(ep.mean)
    const sv = boundary(svm.w)
    return [
      {
        name: 'points',
        x: data.map((d) => d.x[0]),
        y: data.map((d) => d.x[1]),
        group: data.map((d) => (d.y > 0 ? 0 : 1)),
        groupNames: ['class +1', 'class −1'],
      },
      { name: 'Bayes point boundary (EP)', x: bp.x, y: bp.y, emphasis: true },
      { name: 'maximum-margin boundary', x: sv.x, y: sv.y, slot: 2, dashed: true },
    ] as const
  }, [data, ep, svm])

  const handles = useMemo(
    (): Handle[] =>
      data.map((d, i) => ({
        kind: 'point',
        at: d.x,
        label: `point ${i + 1}`,
        onDrag: (p: Vec2) =>
          setData((old) => old.map((o, j) => (j === i ? { ...o, x: [clamp(p[0]), clamp(p[1])] as Vec2 } : o))),
      })),
    [data],
  )

  const overlay = useMemo(() => {
    const n = Math.hypot(ep.mean[0], ep.mean[1]) || 1
    return [
      { name: 'EP mean (Bayes point)', x: [ep.mean[0]], y: [ep.mean[1]], emphasis: true },
      { name: 'exact posterior mean', x: [exact[0]], y: [exact[1]], slot: 1 },
      { name: 'maximum-margin direction', x: [0, svm.w[0] * n], y: [0, svm.w[1] * n], slot: 2 },
    ] as const
  }, [ep, exact, svm])

  const xAxis = useAxis({ label: 'x₁', range: [-3, 3] })
  const yAxis = useAxis({ label: 'x₂', range: [-3, 3], equal: xAxis })
  const xAxis2 = useAxis({ label: 'w₁' })
  const yAxis2 = useAxis({ label: 'w₂' })
  return (
    <Figure
      title="The Bayes point and the maximum-margin classifier"
      state={state}
      caption="Left: six labelled points and two linear boundaries through the origin. Drag any point. Right: the posterior over the weight vector w = (w₁, w₂) under a N(0, I) prior and probit likelihoods with noise β; it is concentrated on the version space, the wedge of weights that classify every point correctly. The Bayes point is the posterior mean, found here by expectation propagation (diamond) and checked against the exact mean on the grid (circle). The dashed line is the boundary with the largest distance to the nearest point, found by a search over directions; it depends only on the points nearest the boundary, while the Bayes point averages over the whole wedge, so the two differ when the wedge is lopsided. A negative margin means no boundary through the origin separates the points."

      readouts={
        <>
          <Readout label="EP vs exact mean, angle" value={`${formatNumber(angle(ep.mean, exact))}°`} />
          <Readout label="Bayes point vs max margin, angle" value={`${formatNumber(angle(ep.mean, svm.w))}°`} />
          <Readout label="margin" value={formatNumber(svm.margin)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points {...dataSeries[0]} />
          <Curve {...dataSeries[1]} />
          <Curve {...dataSeries[2]} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={360}>
          <Raster x={WS} y={WS} z={z} valueLabel={'posterior density (unnormalised)'} />
          <Points {...overlay[0]} live />
          <Points {...overlay[1]} live />
          <Curve {...overlay[2]} live />
        </Plot>
      </div>
    </Figure>
  )
}

const clamp = (v: number) => Math.max(-2.9, Math.min(2.9, v))

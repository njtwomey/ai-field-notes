import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
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
  const beta = useParam(0.2, { min: 0.05, max: 1.5, step: 0.05 })

  const { ep, exact, svm, z } = useMemo(() => {
    const g = posteriorGrid(data, beta.value, WS)
    return { ep: epBpm(data, beta.value, 30), exact: g.mean, svm: maxMarginDirection(data), z: g.z }
  }, [data, beta.value])

  const dataSeries = useMemo((): XYSeries[] => {
    const bp = boundary(ep.mean)
    const sv = boundary(svm.w)
    return [
      {
        name: 'points',
        type: 'scatter',
        x: data.map((d) => d.x[0]),
        y: data.map((d) => d.x[1]),
        group: data.map((d) => (d.y > 0 ? 0 : 1)),
        groupNames: ['class +1', 'class −1'],
      },
      { name: 'Bayes point boundary (EP)', type: 'line', x: bp.x, y: bp.y, emphasis: true },
      { name: 'maximum-margin boundary', type: 'line', x: sv.x, y: sv.y, slot: 2, dashed: true },
    ]
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

  const overlay = useMemo((): HeatmapOverlay[] => {
    const n = Math.hypot(ep.mean[0], ep.mean[1]) || 1
    return [
      { name: 'EP mean (Bayes point)', type: 'scatter', x: [ep.mean[0]], y: [ep.mean[1]], emphasis: true },
      { name: 'exact posterior mean', type: 'scatter', x: [exact[0]], y: [exact[1]], slot: 1 },
      { name: 'maximum-margin direction', type: 'line', x: [0, svm.w[0] * n], y: [0, svm.w[1] * n], slot: 2 },
    ]
  }, [ep, exact, svm])

  return (
    <Interactive
      title="The Bayes point and the maximum-margin classifier"
      caption="Left: six labelled points and two linear boundaries through the origin. Drag any point. Right: the posterior over the weight vector w = (w₁, w₂) under a N(0, I) prior and probit likelihoods with noise β; it is concentrated on the version space, the wedge of weights that classify every point correctly. The Bayes point is the posterior mean, found here by expectation propagation (diamond) and checked against the exact mean on the grid (circle). The dashed line is the boundary with the largest distance to the nearest point, found by a search over directions; it depends only on the points nearest the boundary, while the Bayes point averages over the whole wedge, so the two differ when the wedge is lopsided. A negative margin means no boundary through the origin separates the points."
      controls={<ParamSlider label="label noise β" param={beta} format={(v) => v.toFixed(2)} />}
      readout={
        <>
          <Readout label="EP vs exact mean, angle" value={`${formatNumber(angle(ep.mean, exact))}°`} />
          <Readout label="Bayes point vs max margin, angle" value={`${formatNumber(angle(ep.mean, svm.w))}°`} />
          <Readout label="margin" value={formatNumber(svm.margin)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          series={dataSeries}
          xLabel="x₁"
          yLabel="x₂"
          xRange={[-3, 3]}
          yRange={[-3, 3]}
          equalAspect
          handles={handles}
        />
        <Heatmap
          x={WS}
          y={WS}
          z={z}
          xLabel="w₁"
          yLabel="w₂"
          valueLabel="posterior density (unnormalised)"
          overlay={overlay}
          height={360}
        />
      </div>
    </Interactive>
  )
}

const clamp = (v: number) => Math.max(-2.9, Math.min(2.9, v))

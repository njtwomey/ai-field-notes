import { child, normal, stream } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'
import { useMemo, useState } from 'react'
import { NumberField, Slider, Switch, useParam } from '@lab/controls'
import { Figure, Tex } from '@lab/layout'
import {
  Area,
  Bars,
  Curve,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  Vectors,
  type SegmentsProps,
  type Vec2,
} from '@lab/viz'

const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

/** Several curves on one axis: hover anywhere to read every curve's value at that x. */
export function DerivativeFigure() {
  const omega = useParam(2.5, { min: 0.5, max: 6 })
  const width = useParam(1.5, { min: 0.4, max: 3 })
  const logH = useParam(-1, { min: -4, max: 0, step: 0.1 })
  const { series, worst } = useMemo(() => {
    const [w, s, h] = [omega.value, width.value, 10 ** logH.value]
    const f = (x: number) => Math.sin(w * x) * Math.exp(-(x * x) / (2 * s * s))
    const df = (x: number) => (w * Math.cos(w * x) - (x / (s * s)) * Math.sin(w * x)) * Math.exp(-(x * x) / (2 * s * s))
    const x = grid(-4, 4, 401)
    const fd = x.map((v) => (f(v + h) - f(v - h)) / (2 * h))
    const exact = x.map(df)
    const series = { x, f: x.map(f), exact, fd }
    return { series, worst: Math.max(...exact.map((v, i) => Math.abs(v - fd[i]))) }
  }, [omega.value, width.value, logH.value])
  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'value' })
  return (
    <Figure
      title="Derivatives: hover to read every curve"
      // TODO(5c): purpose taken from the description
      purpose={
        <>
          <Tex>{String.raw`f(x) = \sin(\omega x)\,e^{-x^2/2s^2}`}</Tex>, its exact derivative and a central difference
          with step <Tex>h</Tex>.
        </>
      }
      defaultSize="L"
      controls={
        <>
          <Slider label={<Tex>\omega</Tex>} param={omega} />
          <Slider label={<Tex>s</Tex>} param={width} />
          <Slider label={<Tex>{String.raw`\log_{10} h`}</Tex>} param={logH} />
        </>
      }
      readouts={<Readout label="max |f′ − difference|" value={formatNumber(worst)} />}
      caption="Hover the chart: a pointer line follows the cursor, each curve's nearest point is marked, and the tooltip and the readout below list all three values at that x. Raise h to watch the difference drift from the exact derivative."
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="f(x)" x={series.x} y={series.f} slot={0} />
        <Curve name="f′(x), exact" x={series.x} y={series.exact} slot={1} />
        <Curve name="central difference" x={series.x} y={series.fd} slot={2} dashed />
      </Plot>
    </Figure>
  )
}

const CLUSTERS = ['cluster 1', 'cluster 2', 'cluster 3']

/** Three seeded clusters; the centroids are draggable handles, and points take the colour of their nearest centroid. */
export function ClusterFigure() {
  const spread = useParam(0.6, { min: 0.1, max: 1.5 })
  const [seed, setSeed] = useState(3)
  const [centroids, setCentroids] = useState<Vec2[]>([
    [-2, -1],
    [2, -1],
    [0, 2],
  ])
  const points = useMemo(() => {
    const s = stream(seed)
    const centres: Vec2[] = [
      [-2.2, -1.2],
      [2, -0.8],
      [0.2, 2.2],
    ]
    const x: number[] = []
    const y: number[] = []
    centres.forEach(([cx, cy], k) => {
      const c = child(s, 'cluster', k)
      for (let i = 0; i < 60; i++) {
        x.push(normal(c, cx, spread.value))
        y.push(normal(c, cy, spread.value))
      }
    })
    return { x, y }
  }, [seed, spread.value])
  const { group, wcss } = useMemo(() => {
    const nearest = points.x.map((px, i) => {
      const d = centroids.map(([cx, cy]) => (px - cx) ** 2 + (points.y[i] - cy) ** 2)
      const k = d.indexOf(Math.min(...d))
      return { k, d: d[k] }
    })
    const group = nearest.map((n) => n.k)
    const wcss = nearest.reduce((sum, n) => sum + n.d, 0)
    return { group, wcss }
  }, [points, centroids])
  const x1 = useAxis({ label: 'x₁' })
  const x2 = useAxis({ label: 'x₂', equal: x1 })
  return (
    <Figure
      title="Scatter groups with draggable centroids"
      // TODO(5c): purpose taken from the description
      purpose="Groups get a fixed slot and a marker shape; centroids are emphasised in ink and are handles."
      controls={
        <>
          <Slider label="spread" param={spread} />
          <NumberField label="seed" value={seed} onChange={setSeed} min={0} max={9999} />
        </>
      }
      readouts={<Readout label="within-cluster sum of squares" value={formatNumber(wcss)} />}
      caption="Drag a centroid (the nearest one within reach is grabbed): points recolour by their nearest centroid. Equal aspect: the chart keeps its frame's size and widens one axis so a unit is the same length on both."
    >
      <Plot x={x1} y={x2}>
        <Points name="points" x={points.x} y={points.y} group={group} groupNames={CLUSTERS} />
        {centroids.map((at, k) => (
          <Handle
            key={k}
            kind="point"
            at={at}
            onDrag={(p: Vec2) => setCentroids((cs) => cs.map((c, j) => (j === k ? p : c)))}
          />
        ))}
      </Plot>
    </Figure>
  )
}

/** Bars, an area and a dashed reference line, with a threshold bound to both a slider and a vertical handle. */
export function HistogramFigure() {
  const threshold = useParam(1.2, { min: -3, max: 3.5, step: 0.05 })
  const draws = useMemo(() => {
    const s = stream('histogram')
    return Array.from({ length: 4000 }, () => normal(s, 0, 1))
  }, [])
  const { series, empirical } = useMemo(() => {
    const edges = grid(-4, 4, 41)
    const width = edges[1] - edges[0]
    const counts = new Array(edges.length - 1).fill(0)
    for (const d of draws) {
      const i = Math.floor((d + 4) / width)
      if (i >= 0 && i < counts.length) counts[i]++
    }
    const centres = counts.map((_, i) => edges[i] + width / 2)
    const x = grid(-4, 4, 321)
    const pdf = (v: number) => Math.exp((-v * v) / 2) / Math.sqrt(2 * Math.PI)
    const tail = x.filter((v) => v >= threshold.value)
    const series = {
      centres,
      edges,
      heights: counts.map((c) => c / (draws.length * width)),
      tail,
      tailY: tail.map(pdf),
      x,
      pdf: x.map(pdf),
    }
    return { series, empirical: draws.filter((d) => d >= threshold.value).length / draws.length }
  }, [draws, threshold.value])
  const xAxis = useAxis({ label: 'x', range: [-4, 4] })
  const yAxis = useAxis({ label: 'density' })
  return (
    <Figure
      title="Histogram, density and a draggable threshold"
      // TODO(5c): purpose taken from the description
      purpose="4000 standard normal draws (aifn/random, seeded), their histogram, the density and the tail beyond t."
      controls={<Slider label="threshold t" param={threshold} />}
      readouts={
        <>
          <Readout label="empirical tail" value={formatNumber(empirical)} />
          <Readout label="1 − Φ(t)" value={formatNumber(1 - normalCdf(threshold.value))} />
        </>
      }
      caption="Drag the dashed line or use the slider: both write the same parameter. The tooltip lists the bar, the tail and the density at the hovered x."
    >
      <Plot x={xAxis} y={yAxis}>
        <Bars name="histogram" x={series.centres} y={series.heights} edges={series.edges} slot={0} />
        <Area name="tail P(X ≥ t)" x={series.tail} y={series.tailY} slot={1} />
        <Curve name="normal density" x={series.x} y={series.pdf} emphasis dashed />
        <Handle kind="x" at={threshold.value} label="t" onDrag={threshold.set} />
      </Plot>
    </Figure>
  )
}

/** Values over several decades on log axes; zoom and pan work in log space. */
export function LogFigure() {
  const [xLog, setXLog] = useState(false)
  const [yLog, setYLog] = useState(true)
  const series = useMemo(() => {
    const t = Array.from({ length: 2000 }, (_, i) => i + 1)
    const s = stream('loss')
    return [0.002, 0.01, 0.05].map((rate, k) => {
      const c = child(s, k)
      return {
        name: `learning rate ${rate}`,
        x: t,
        y: t.map((i) => (10 * Math.exp(-rate * i) + 1e-4 / rate) * Math.exp(0.15 * normal(c, 0, 1))),
        slot: k,
      }
    })
  }, [])
  const xAxis = useAxis({ label: 'iteration', log: xLog })
  const yAxis = useAxis({ label: 'loss', log: yLog })
  return (
    <Figure
      title="Log axes: zoom and pan in log space"
      // TODO(5c): purpose taken from the description
      purpose="Three loss curves falling over several orders of magnitude."
      controls={
        <>
          <Switch label="log x" checked={xLog} onChange={setXLog} />
          <Switch label="log y" checked={yLog} onChange={setYLog} />
        </>
      }
      caption="Use the toolbar: per-axis pan (arrows), zoom out and in (− +), typed ranges, and for both axes zoom and fit. Pinch or Ctrl/⌘-scroll over the plot zooms about the pointer. On a log axis a zoom keeps decades proportional."
    >
      <Plot x={xAxis} y={yAxis}>
        {series.map((c) => (
          <Curve key={c.name} name={c.name} x={c.x} y={c.y} slot={c.slot} />
        ))}
      </Plot>
    </Figure>
  )
}

const SIDES = ['w·x < 0', 'w·x ≥ 0']

/** Arrows and segments with equal aspect: a draggable normal vector and each point's distance to its boundary. */
export function VectorFigure() {
  const [tip, setTip] = useState<Vec2>([1, 1.5])
  const points = useMemo(() => {
    const s = stream('vectors')
    return Array.from({ length: 24 }, (_, i): Vec2 => {
      const c = child(s, i)
      return [normal(c, 0, 1.4), normal(c, 0, 1.4)]
    })
  }, [])
  const { series, segments, angle } = useMemo(() => {
    const [wx, wy] = tip
    const norm = Math.hypot(wx, wy) || 1
    const [ux, uy] = [wx / norm, wy / norm]
    // The boundary w·x = 0 runs along (−uy, ux); each segment drops a point onto it.
    const segments: SegmentsProps['segments'] = points.map(([px, py]) => {
      const d = px * ux + py * uy
      return { from: [px, py], to: [px - d * ux, py - d * uy] }
    })
    const side = points.map(([px, py]) => (px * ux + py * uy >= 0 ? 1 : 0))
    const series = {
      boundary: { x: [-4 * -uy, 4 * -uy], y: [-4 * ux, 4 * ux] },
      points: { x: points.map((p) => p[0]), y: points.map((p) => p[1]), group: side },
    }
    return { series, segments, angle: (Math.atan2(wy, wx) * 180) / Math.PI }
  }, [tip, points])
  const vectors = useMemo(() => [{ from: [0, 0] as Vec2, to: tip, label: 'w' }], [tip])
  const x1 = useAxis({ label: 'x₁', range: [-4, 4] })
  const x2 = useAxis({ label: 'x₂', range: [-4, 4], equal: x1 })
  return (
    <Figure
      title="Vectors, segments and equal aspect"
      // TODO(5c): purpose taken from the description
      purpose="A weight vector w, its boundary, and each point's perpendicular drop onto it."
      readouts={
        <>
          <Readout label="angle of w" value={`${formatNumber(angle)}°`} />
          <Readout label="‖w‖" value={formatNumber(Math.hypot(...tip))} />
        </>
      }
      caption="Drag the tip of w. With equal aspect the arrow stays perpendicular to the boundary whatever the frame's shape."
      defaultSize="S"
    >
      <Plot x={x1} y={x2}>
        <Curve name="boundary w·x = 0" x={series.boundary.x} y={series.boundary.y} muted dashed />
        <Segments segments={segments} />
        <Points name="points" {...series.points} groupNames={SIDES} />
        <Vectors vectors={vectors} />
        <Handle kind="point" at={tip} label="w" onDrag={setTip} />
      </Plot>
    </Figure>
  )
}

const ORIGIN = [0]

/** A vector that can run off the plot: its shaft is clipped at the edge, with a chevron in place of the arrowhead. */
export function ClippedVectorFigure() {
  const length = useParam(4, { min: 0.5, max: 10 })
  const angle = useParam(30, { min: -180, max: 180, step: 1 })
  const [base, setBase] = useState<Vec2>([-1, -1])
  const vectors = useMemo(() => {
    const t = (angle.value * Math.PI) / 180
    const tip: Vec2 = [base[0] + length.value * Math.cos(t), base[1] + length.value * Math.sin(t)]
    // A second, fixed vector, a quarter turn round: it too leaves the box when long enough.
    const turned: Vec2 = [base[0] - length.value * Math.sin(t), base[1] + length.value * Math.cos(t)]
    return [
      { from: base, to: tip, label: 'v' },
      { from: base, to: turned, slot: 1, label: 'v⊥' },
    ]
  }, [base, length.value, angle.value])
  const x1 = useAxis({ label: 'x₁', range: [-3, 3] })
  const x2 = useAxis({ label: 'x₂', range: [-3, 3], equal: x1 })
  return (
    <Figure
      title="Vectors that leave the plot"
      // TODO(5c): purpose taken from the description
      purpose="Axes fixed to [−3, 3]; lengthen the vectors or drag their base until the tips leave the box."
      defaultSize="S"
      controls={
        <>
          <Slider label="length" param={length} />
          <Slider label="angle (degrees)" param={angle} />
        </>
      }
      caption="A tip outside the box is cut at the edge: the shaft ends there with a small chevron pointing the way the vector goes, and the label moves to that point. Zoom in (the toolbar or a pinch) to clip further; a vector wholly outside draws nothing."
    >
      <Plot x={x1} y={x2}>
        <Points name="origin" x={ORIGIN} y={ORIGIN} muted />
        <Vectors vectors={vectors} />
        <Handle kind="point" at={base} label="base" onDrag={setBase} />
      </Plot>
    </Figure>
  )
}

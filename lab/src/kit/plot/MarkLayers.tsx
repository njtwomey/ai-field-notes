/** UI-kit figures for the basic layers: Curve, Points, Bars, Area, SignedArea, Segments, Vectors, Rug, Annotation, Handle. */
import { child, normal, stream } from 'aifn-compute/foundation/random'
import { Normal } from 'aifn-compute/probability/distributions'
import { normalCdf } from 'aifn-compute/numerics/special'
import { useMemo, useState } from 'react'
import { Select, Slider, Switch, useParam } from 'aifn-render/controls'
import { seriesColor, useTheme } from 'aifn-render/design'
import { Figure } from 'aifn-render/layout'
import {
  Annotation,
  Area,
  Bars,
  Curve,
  Density,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Rug,
  Segments,
  SignedArea,
  signedArea,
  useAxis,
  Vectors,
  type Vec2,
} from 'aifn-render/viz'
import { grid } from './data'

// ── Curve ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A function, its derivative and a live tangent at a draggable x₀; the axes hold while ω changes. */
export function CurveFigure() {
  const omega = useParam(1.5, { min: 0.5, max: 4 })
  const x0 = useParam(1, { min: -4, max: 4 })
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'value', hold: 'union' })
  const { xs, f, df } = useMemo(() => {
    const w = omega.value
    const xs = grid(-4, 4, 401)
    const f = xs.map((v) => Math.sin(w * v) * Math.exp(-(v * v) / 8))
    const df = xs.map((v) => (w * Math.cos(w * v) - (v / 4) * Math.sin(w * v)) * Math.exp(-(v * v) / 8))
    return { xs, f, df }
  }, [omega.value])
  const w = omega.value
  const at = x0.value
  const f0 = Math.sin(w * at) * Math.exp(-(at * at) / 8)
  const slope = (w * Math.cos(w * at) - (at / 4) * Math.sin(w * at)) * Math.exp(-(at * at) / 8)
  return (
    <Figure
      title="Curve: a tangent that follows a handle"
      // TODO(5c): purpose taken from the description
      purpose="f(x) = sin(ωx)·e^(−x²/8) and f′ as Curve layers; the tangent at x₀ is a live Curve, sent as a patch."
      defaultSize="L"
      controls={
        <>
          <Slider label="ω" param={omega} />
          <Slider label="x₀" param={x0} />
        </>
      }
      readouts={
        <>
          <Readout label="f(x₀)" value={formatNumber(f0)} />
          <Readout label="f′(x₀)" value={formatNumber(slope)} />
        </>
      }
      caption="Drag x₀ on the chart: only the tangent and the guide line move (one patch per frame). The y axis is held with `hold: 'union'`: raising ω reshapes the curves on a fixed scale, which only ever grows."
    >
      <Plot x={x} y={y}>
        <Curve name="f(x)" x={xs} y={f} />
        <Curve name="f′(x)" x={xs} y={df} dashed />
        <Curve name="tangent" x={[at - 1.2, at + 1.2]} y={[f0 - 1.2 * slope, f0 + 1.2 * slope]} slot={2} live />
        <Handle kind="x" at={at} label="x₀" onDrag={x0.set} />
      </Plot>
    </Figure>
  )
}

// ── Points ───────────────────────────────────────────────────────────────────────────────────────────────────────────

const CLASSES = ['setosa', 'versicolor', 'virginica']

/** Three seeded classes with draggable centroids; colour is the class, shape is a second variable. */
export function PointsFigure() {
  const [centres, setCentres] = useState<Vec2[]>([
    [-2, -1],
    [2, -1],
    [0, 2],
  ])
  const [split, setSplit] = useState(true)
  const data = useMemo(() => {
    const s = stream('points-kit')
    const x: number[] = []
    const y: number[] = []
    const group: number[] = []
    const shape: number[] = []
    for (let k = 0; k < 3; k++)
      for (let i = 0; i < 60; i++) {
        const c = child(s, k, i)
        x.push(normal(c, 0, 0.7))
        y.push(normal(child(c, 'y'), 0, 0.7))
        group.push(k)
        shape.push(i % 2)
      }
    return { x, y, group, shape }
  }, [])
  const shifted = useMemo(
    () => ({
      x: data.x.map((v, i) => v + centres[data.group[i]][0]),
      y: data.y.map((v, i) => v + centres[data.group[i]][1]),
    }),
    [data, centres],
  )
  const x = useAxis({ label: 'x₁', hold: 'union' })
  const y = useAxis({ label: 'x₂', hold: 'union', equal: x })
  return (
    <Figure
      title="Points: classes, marker shapes and draggable centroids"
      // TODO(5c): purpose taken from the description
      purpose="Class k takes slot k and, by default, marker shape k; `shape` gives a second variable its own shapes."
      defaultSize="L"
      controls={<Switch label="shape by fold (○ □)" checked={split} onChange={setSplit} />}
      caption="Drag a centroid (the ink marks): its class follows. Units are equal (y is linked to x), so the clusters keep their shape. With the switch on, circles and squares mark two folds within each class; colour stays the class."
    >
      <Plot x={x} y={y}>
        <Points
          name="points"
          x={shifted.x}
          y={shifted.y}
          group={data.group}
          groupNames={CLASSES}
          shape={split ? data.shape : undefined}
        />
        <Points name="centroids" x={centres.map((c) => c[0])} y={centres.map((c) => c[1])} emphasis live />
        {centres.map((c, k) => (
          <Handle key={k} kind="point" at={c} onDrag={(p) => setCentres((cs) => cs.map((v, j) => (j === k ? p : v)))} />
        ))}
      </Plot>
    </Figure>
  )
}

/** The Hénon attractor: 30 000 points in the dense mode. */
export function DensePointsFigure() {
  const a = useParam(1.4, { min: 1.0, max: 1.42, step: 0.01 })
  const pts = useMemo(() => {
    const n = 30000
    const x = new Float64Array(n)
    const y = new Float64Array(n)
    let [u, v] = [0.1, 0.1]
    for (let i = 0; i < n + 100; i++) {
      ;[u, v] = [1 - a.value * u * u + v, 0.3 * u]
      if (!Number.isFinite(u) || Math.abs(u) > 1e6) [u, v] = [0.1, 0.1]
      if (i >= 100) [x[i - 100], y[i - 100]] = [u, v]
    }
    return { x, y }
  }, [a.value])
  const x = useAxis({ label: 'x', range: [-1.6, 1.6] })
  const y = useAxis({ label: 'y', range: [-0.45, 0.45] })
  return (
    <Figure
      title="Points: a dense attractor"
      // TODO(5c): purpose taken from the description
      purpose="The Hénon map xₙ₊₁ = 1 − a xₙ² + yₙ, yₙ₊₁ = 0.3 xₙ: 30 000 iterates as tiny unoutlined dots (dense mode, canvas)."
      controls={<Slider label="a" param={a} />}
      caption="`dense` draws thousands of small marks as one path; the Plot switches to canvas. Lower a below about 1.06 and the orbit settles on a cycle."
    >
      <Plot x={x} y={y}>
        <Points name="orbit" x={pts.x} y={pts.y} dense />
      </Plot>
    </Figure>
  )
}

// ── Bars ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

const FRUIT = ['apple', 'banana', 'cherry', 'damson', 'elder']

/** Counts per category on a categorical x axis, and the same as horizontal bars on a categorical y axis. */
export function BarsFigure() {
  const counts = [12, 7, 15, 4, 9]
  const { resolved: mode } = useTheme()
  const x = useAxis({ label: 'fruit', categories: FRUIT })
  const y = useAxis({ label: 'count' })
  const k = useAxis({ label: 'count' })
  const cat = useAxis({ label: 'fruit', categories: FRUIT })
  const idx = FRUIT.map((_, i) => i)
  return (
    <>
      <Figure
        title="Bars: a categorical x axis"
        // TODO(5c): purpose taken from the description
        purpose="`useAxis({ categories })` places category k at position k, labelled by name; bars are data-unit rectangles."
        caption="Bars span exactly 0.8 of a category; the axis has no zoom (categories do not zoom)."
      >
        <Plot x={x} y={y}>
          <Bars name="count" x={idx} y={counts} />
        </Plot>
      </Figure>
      <Figure
        title="Bars: horizontal, coloured by class"
        // TODO(5c): purpose taken from the description
        purpose="orient y lays bars along y; colors gives each bar its class colour (alternate classes here)."
      >
        <Plot x={k} y={cat}>
          <Bars name="count" x={idx} y={counts} orient="y" colors={idx.map((i) => seriesColor(mode, i % 2))} />
        </Plot>
      </Figure>
    </>
  )
}

// ── Area ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A band between two curves (mean ± k·sd of a random walk's position) and its mean. */
export function AreaFigure() {
  const k = useParam(1, { min: 0.5, max: 3 })
  const t = useMemo(() => grid(0, 10, 101), [])
  const sd = useMemo(() => t.map((v) => Math.sqrt(v)), [t])
  const mean = useMemo(() => t.map((v) => 0.3 * v), [t])
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'position', hold: 'union' })
  return (
    <Figure
      title="Area: a band between two curves"
      // TODO(5c): purpose taken from the description
      purpose="Area with an array `base`: the region between mean − k·sd and mean + k·sd of a drifting random walk."
      controls={<Slider label="k" param={k} />}
    >
      <Plot x={x} y={y}>
        <Area
          name={`mean ± ${k.value} sd`}
          x={t}
          y={mean.map((m, i) => m + k.value * sd[i])}
          base={mean.map((m, i) => m - k.value * sd[i])}
          line={false}
        />
        <Curve name="mean" x={t} y={mean} emphasis />
      </Plot>
    </Figure>
  )
}

// ── SignedArea ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** KL(p‖q) as the net signed area under p log(p/q). */
export function SignedAreaFigure() {
  const mu = useParam(1, { min: -2, max: 2 })
  const sd = useParam(1.5, { min: 0.4, max: 3 })
  const { xs, integrand } = useMemo(() => {
    const p = Normal(0, 1)
    const q = Normal(mu.value, sd.value)
    const xs = grid(-5, 5, 401)
    return { xs, integrand: xs.map((v) => p.prob(v) * (p.logProb(v) - q.logProb(v))) }
  }, [mu.value, sd.value])
  const area = signedArea(xs, integrand)
  const exact = Math.log(sd.value) + (1 + mu.value ** 2) / (2 * sd.value ** 2) - 0.5
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'p log(p/q)', hold: 'union' })
  return (
    <Figure
      title="SignedArea: KL divergence as a net area"
      // TODO(5c): purpose taken from the description
      purpose="KL(p‖q) = ∫ p log(p/q) dx for p = N(0, 1) and q = N(μ, σ): positive where p > q, negative where q > p."
      controls={
        <>
          <Slider label="μ (q)" param={mu} />
          <Slider label="σ (q)" param={sd} />
        </>
      }
      readouts={
        <>
          <Readout label="net area (trapezoid)" value={formatNumber(area.net)} />
          <Readout label="KL, closed form" value={formatNumber(exact)} />
        </>
      }
      caption="The two signs take the ends of the diverging scale; the label in the corner is the layer's own net area, which matches the closed form."
    >
      <Plot x={x} y={y}>
        <SignedArea name="p log(p/q)" x={xs} y={integrand} label="KL" />
      </Plot>
    </Figure>
  )
}

// ── Segments ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Residuals from points to a line whose slope is a slider. */
export function SegmentsFigure() {
  const slope = useParam(0.8, { min: -1, max: 2 })
  const data = useMemo(() => {
    const s = stream('segments-kit')
    const x = grid(0, 10, 25)
    return { x, y: x.map((v, i) => 0.5 + 0.7 * v + normal(child(s, i), 0, 1)) }
  }, [])
  const fit = data.x.map((v) => 0.5 + slope.value * v)
  const sse = data.y.reduce((a, v, i) => a + (v - fit[i]) ** 2, 0)
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', hold: 'initial' })
  return (
    <Figure
      title="Segments: residuals to a line"
      // TODO(5c): purpose taken from the description
      purpose="Segments draws thin muted lines: here from each point to the line y = 0.5 + b·x."
      controls={<Slider label="b" param={slope} />}
      readouts={<Readout label="sum of squares" value={formatNumber(sse)} />}
    >
      <Plot x={x} y={y}>
        <Segments segments={data.x.map((v, i) => ({ from: [v, data.y[i]] as const, to: [v, fit[i]] as const }))} />
        <Points name="data" x={data.x} y={data.y} />
        <Curve name="line" x={[0, 10]} y={[fit[0], fit[fit.length - 1]]} emphasis />
      </Plot>
    </Figure>
  )
}

// ── Vectors ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A vector with a draggable tip, its image under a matrix, clipped where it leaves the plot. */
export function VectorsFigure() {
  const [tip, setTip] = useState<Vec2>([1.5, 1])
  const angle = useParam(60, { min: 0, max: 180, step: 1 })
  const scale = useParam(1.6, { min: 0.5, max: 3 })
  const r = (angle.value * Math.PI) / 180
  const image: Vec2 = [
    scale.value * (Math.cos(r) * tip[0] - Math.sin(r) * tip[1]),
    scale.value * (Math.sin(r) * tip[0] + Math.cos(r) * tip[1]),
  ]
  const x = useAxis({ label: 'x₁', range: [-3, 3] })
  const y = useAxis({ label: 'x₂', range: [-3, 3], equal: x })
  return (
    <Figure
      title="Vectors: a draggable tip and its image"
      // TODO(5c): purpose taken from the description
      purpose="v (ink) and Av for A = s·R(θ), a rotation and scaling, in slot 1; equal units, so angles are true."
      controls={
        <>
          <Slider label="θ (degrees)" param={angle} />
          <Slider label="s" param={scale} />
        </>
      }
      caption="Drag the tip of v. When Av runs off the plot it is clipped at the edge and ends in a chevron pointing the way it goes."
    >
      <Plot x={x} y={y}>
        <Vectors
          vectors={[
            { from: [0, 0], to: tip, label: 'v' },
            { from: [0, 0], to: image, slot: 1, label: 'Av' },
          ]}
          live
        />
        <Handle kind="point" at={tip} onDrag={setTip} />
      </Plot>
    </Figure>
  )
}

// ── Rug ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Draws under the density they come from. */
export function RugFigure() {
  const n = useParam(40, { min: 5, max: 200, step: 5 })
  const draws = useMemo(() => {
    const s = stream('rug-kit')
    return Array.from({ length: n.value }, (_, i) => normal(child(s, i), 0, 1))
  }, [n.value])
  const dist = useMemo(() => Normal(0, 1), [])
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'density' })
  return (
    <Figure
      title="Rug: draws along the axis"
      // TODO(5c): purpose taken from the description
      purpose="A tick per draw along the bottom edge, under the N(0, 1) density they are drawn from."
      controls={<Slider label="draws" param={n} />}
    >
      <Plot x={x} y={y}>
        <Density dist={dist} name="N(0, 1)" fill={0.12} />
        <Rug values={draws} slot={0} />
      </Plot>
    </Figure>
  )
}

// ── Annotation ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A labelled maximum, a vertical line at the mean and a horizontal line at half height. */
export function AnnotationFigure() {
  const xs = useMemo(() => grid(0, 8, 321), [])
  const ys = useMemo(() => xs.map((v) => v * Math.exp(-v)), [xs])
  const x = useAxis({ label: 't' })
  const y = useAxis({ label: 'f(t)' })
  return (
    <Figure
      title="Annotation: points and lines with labels"
      // TODO(5c): purpose taken from the description
      purpose="f(t) = t·e^(−t): its maximum at t = 1 (a labelled point), its mean t = 2 and half its height."
    >
      <Plot x={x} y={y}>
        <Curve name="f(t)" x={xs} y={ys} />
        <Annotation at={[1, Math.exp(-1)]} text="max at t = 1" />
        <Annotation x={2} text="mean" />
        <Annotation y={Math.exp(-1) / 2} text="half max" />
      </Plot>
    </Figure>
  )
}

// ── Handle ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A cdf with an x handle (the threshold) and a y handle (the probability), each a natural place on the chart. */
export function HandleFigure() {
  const [threshold, setThreshold] = useState(0.5)
  const [p, setP] = useState(0.9)
  const [kind, setKind] = useState<'x' | 'y'>('x')
  const xs = useMemo(() => grid(-4, 4, 321), [])
  const cdf = useMemo(() => xs.map((v) => normalCdf(v)), [xs])
  const q = Normal(0, 1).quantile(p) as number
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'Φ(x)', range: [0, 1] })
  return (
    <Figure
      title="Handle: a threshold and a probability"
      // TODO(5c): purpose taken from the description
      purpose="The standard normal cdf Φ. An x handle sets a threshold and reads Φ there; a y handle sets a probability and reads its quantile."
      controls={
        <Select
          label="handle"
          value={kind}
          onChange={(v) => setKind(v as 'x' | 'y')}
          options={[
            { value: 'x', label: 'threshold (x)' },
            { value: 'y', label: 'probability (y)' },
          ]}
        />
      }
      readouts={
        kind === 'x' ? (
          <Readout label="Φ(threshold)" value={formatNumber(normalCdf(threshold))} />
        ) : (
          <Readout label="quantile" value={formatNumber(q)} />
        )
      }
      caption="Press anywhere on the plot to move the handle there; the axes hold still while it moves."
    >
      <Plot x={x} y={y}>
        <Curve name="Φ(x)" x={xs} y={cdf} />
        {kind === 'x' ? (
          <>
            <Handle
              kind="x"
              at={threshold}
              label="threshold"
              onDrag={(v) => setThreshold(Math.max(-4, Math.min(4, v)))}
            />
            <Points name="Φ(threshold)" x={[threshold]} y={[normalCdf(threshold)]} emphasis live />
          </>
        ) : (
          <>
            <Handle kind="y" at={p} label="p" onDrag={(v) => setP(Math.max(0.001, Math.min(0.999, v)))} />
            <Points name="quantile" x={[q]} y={[p]} emphasis live />
          </>
        )}
      </Plot>
    </Figure>
  )
}

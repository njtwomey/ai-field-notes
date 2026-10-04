/** UI-kit figures for the probability and field layers: Histogram, Density, Mass, SupportBand, Raster, Contours. */
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Beta, Binomial, Gamma, Normal, Poisson, Uniform, type Univariate } from 'aifn/probability/distributions'
import { useMemo, useState } from 'react'
import { Select, Slider, Switch, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import {
  Contours,
  Curve,
  Density,
  formatNumber,
  Handle,
  Histogram,
  Mass,
  Plot,
  Points,
  Raster,
  Readout,
  SupportBand,
  supportOf,
  useAxis,
  Vectors,
  type Vec2,
} from '@lab/viz'
import { BUMPS, bump, density2, grid } from './data'

const draws = (d: Univariate, seed: string, n: number) => toFlat(d.sample(stream(seed), { shape: [n] }) as Tensor)

// ── Histogram ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Gamma draws binned by aifn's histogram, under the density they are drawn from. */
export function HistogramFigure() {
  const shape = useParam(2, { min: 0.5, max: 8 })
  const [n, setN] = useState('2000')
  const [counts, setCounts] = useState(false)
  const dist = useMemo(() => Gamma(shape.value, 1), [shape.value])
  const values = useMemo(() => draws(dist, 'histogram-kit', Number(n)), [dist, n])
  const x = useAxis({ label: 'x', support: supportOf(dist), hold: 'union', key: counts })
  const y = useAxis({ label: counts ? 'count' : 'density', key: counts })
  return (
    <Figure
      title="Histogram: draws under their density"
      // TODO(5c): purpose taken from the description
      purpose="Gamma(k, 1) draws binned by aifn/probability/stats (Freedman–Diaconis), with the density they come from."
      controls={
        <>
          <Slider label="shape k" param={shape} />
          <Select label="draws" value={n} onChange={setN} options={['200', '2000', '20000']} />
          <Switch label="counts" checked={counts} onChange={setCounts} />
        </>
      }
      caption="The bars are outlined in their colour so they read at any fill. The x axis is support-bound: it starts at 0, the Gamma's lower end, and its unbounded end fits the data."
    >
      <Plot x={x} y={y}>
        <Histogram name="draws" values={values} normalize={counts ? 'count' : 'density'} />
        {!counts && <Density dist={dist} name={`Gamma(${formatNumber(shape.value)}, 1)`} emphasis />}
      </Plot>
    </Figure>
  )
}

// ── Density ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Three Beta densities on the support-bound unit interval, with a held density axis. */
export function DensityFigure() {
  const a = useParam(2, { min: 0.3, max: 8 })
  const b = useParam(5, { min: 0.3, max: 8 })
  const family = useMemo(
    () => [Beta(a.value, b.value), Beta(b.value, a.value), Beta(a.value, a.value)],
    [a.value, b.value],
  )
  const x = useAxis({ label: 'x', support: { lower: 0, upper: 1 } })
  const y = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="Density: Beta densities on the unit interval"
      // TODO(5c): purpose taken from the description
      purpose="Beta(a, b), its mirror Beta(b, a) and the symmetric Beta(a, a), each evaluated by its aifn distribution object."
      controls={
        <>
          <Slider label="a" param={a} />
          <Slider label="b" param={b} />
        </>
      }
      caption="The x axis is the support [0, 1] exactly. The density axis is held with union: it grows when a density peaks higher and never jumps back. Below a = 1 the density is infinite at 0; the curve stops short of the end rather than spiking the axis."
    >
      <Plot x={x} y={y}>
        <Density dist={family[0]} name="Beta(a, b)" fill={0.12} />
        <Density dist={family[1]} name="Beta(b, a)" />
        <Density dist={family[2]} name="Beta(a, a)" dashed />
      </Plot>
    </Figure>
  )
}

// ── Mass ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Poisson and binomial mass functions with equal means. */
export function MassFigure() {
  const mean = useParam(4, { min: 0.5, max: 12 })
  const n = useParam(20, { min: 12, max: 60, step: 1 })
  const poisson = useMemo(() => Poisson(mean.value), [mean.value])
  const binomial = useMemo(() => Binomial(n.value, mean.value / n.value), [mean.value, n.value])
  const x = useAxis({ label: 'k', range: [-0.5, 25.5] })
  const y = useAxis({ label: 'P(K = k)', hold: 'union' })
  return (
    <Figure
      title="Mass: a Poisson and a binomial with one mean"
      // TODO(5c): purpose taken from the description
      purpose="Poisson(λ) and Binomial(n, λ/n) as bars at each integer; the binomial tends to the Poisson as n grows."
      controls={
        <>
          <Slider label="λ" param={mean} />
          <Slider label="n" param={n} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Mass dist={poisson} name="Poisson(λ)" width={0.4} range={[0, 25]} />
        <Mass dist={binomial} name="Binomial(n, λ/n)" width={0.4} opacity={0.6} range={[0, 25]} />
      </Plot>
    </Figure>
  )
}

// ── SupportBand ──────────────────────────────────────────────────────────────────────────────────────────────────────

const SUPPORTS: Record<string, () => Univariate> = {
  'Normal(0, 1)': () => Normal(0, 1),
  'Gamma(2, 1)': () => Gamma(2, 1),
  'Beta(2, 3)': () => Beta(2, 3),
  'Uniform(−1, 2)': () => Uniform(-1, 2),
}

/** Where each distribution lives: a band along the axis, open or closed ends, the outside shaded. */
export function SupportBandFigure() {
  const [name, setName] = useState('Gamma(2, 1)')
  const dist = useMemo(() => SUPPORTS[name](), [name])
  const x = useAxis({ label: 'x', range: [-2.5, 6] })
  const y = useAxis({ label: 'density', key: name })
  return (
    <Figure
      title="SupportBand: where a distribution lives"
      // TODO(5c): purpose taken from the description
      purpose="The support as a band along the bottom edge: a filled dot at a closed end, a hollow dot at an open one, off the plot where it is unbounded; the outside is shaded."
      controls={<Select label="distribution" value={name} onChange={setName} options={Object.keys(SUPPORTS)} />}
    >
      <Plot x={x} y={y}>
        <SupportBand dist={dist} slot={0} />
        <Density dist={dist} name={name} />
      </Plot>
    </Figure>
  )
}

// ── Raster and Contours ──────────────────────────────────────────────────────────────────────────────────────────────

function gradient(x: number, y: number): Vec2 {
  let [gx, gy] = [0, 0]
  for (const b of BUMPS) {
    const v = bump(b, x, y) / b.width ** 2
    gx -= v * (x - b.at[0])
    gy -= v * (y - b.at[1])
  }
  return [gx, gy]
}

const BUMP_NAMES = ['bump 1', 'bump 2', 'bump 3']

/** A field as a raster: sequential, diverging (symmetric, pale midpoint) or categorical; a draggable start. */
export function RasterFigure() {
  const [scale, setScale] = useState<'sequential' | 'diverging' | 'categorical'>('sequential')
  const [contours, setContours] = useState(true)
  const [start, setStart] = useState<Vec2>([2.2, 2.4])
  const rate = useParam(0.6, { min: 0.05, max: 2 })
  const xs = useMemo(() => grid(-3.5, 3.5, 71), [])
  const fields = useMemo(() => {
    const dens = xs.map((yv) => xs.map((xv) => density2(xv, yv)))
    const mean = dens.flat().reduce((a, b) => a + b, 0) / dens.flat().length
    return {
      sequential: dens,
      diverging: dens.map((row) => row.map((v) => v - mean)),
      categorical: xs.map((yv) =>
        xs.map((xv) => {
          const v = BUMPS.map((b) => bump(b, xv, yv))
          const top = Math.max(...v)
          return top < 0.05 ? -1 : v.indexOf(top)
        }),
      ),
    }
  }, [xs])
  const path = useMemo(() => {
    const px = [start[0]]
    const py = [start[1]]
    for (let i = 0; i < 40; i++) {
      const [gx, gy] = gradient(px[i], py[i])
      px.push(px[i] + rate.value * gx)
      py.push(py[i] + rate.value * gy)
    }
    return { x: px, y: py }
  }, [start, rate.value])
  const [gx, gy] = gradient(...start)
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y', equal: x })
  return (
    <Figure
      title="Raster: a field with contours and a draggable start"
      // TODO(5c): purpose taken from the description
      purpose="A mixture of three bumps on a 71 × 71 grid, drawn once into a canvas image, with a gradient-ascent path from a draggable start."
      defaultSize="L"
      controls={
        <>
          <Select
            label="scale"
            value={scale}
            onChange={(v) => setScale(v as typeof scale)}
            options={['sequential', 'diverging', 'categorical']}
          />
          <Slider label="step size" param={rate} />
          <Switch label="contours" checked={contours} onChange={setContours} />
        </>
      }
      readouts={<Readout label="density at start" value={formatNumber(density2(...start))} />}
      caption="Drag the start: the path, the gradient arrow and the handle are live layers, so the image is never redrawn. Diverging is symmetric about zero by default: zero is the pale midpoint, the extremes are saturated. Equal units, with the plot area fitted to the grid (no padding)."
    >
      <Plot x={x} y={y}>
        <Raster x={xs} y={xs} z={fields[scale]} scale={scale} valueLabel="density" categoryNames={BUMP_NAMES} />
        {contours && <Contours x={xs} y={xs} z={fields.sequential} levels={[0.2, 0.4, 0.6, 0.8]} />}
        <Curve name="gradient ascent" x={path.x} y={path.y} showPoints slot={scale === 'sequential' ? 1 : 4} live />
        <Vectors vectors={[{ from: start, to: [start[0] + gx, start[1] + gy] }]} live />
        <Handle kind="point" at={start} onDrag={setStart} />
      </Plot>
    </Figure>
  )
}

/** Contours of the Rosenbrock function with gradient-descent iterates. */
export function ContoursFigure() {
  const levels = useParam(8, { min: 3, max: 16, step: 1 })
  const xs = useMemo(() => grid(-2, 2, 121), [])
  const ys = useMemo(() => grid(-1, 3, 121), [])
  const z = useMemo(() => ys.map((b) => xs.map((a) => Math.log1p((1 - a) ** 2 + 100 * (b - a * a) ** 2))), [xs, ys])
  const iterates = useMemo(() => {
    let [a, b] = [-1.5, 2.5]
    const px = [a]
    const py = [b]
    for (let i = 0; i < 400; i++) {
      const ga = -2 * (1 - a) - 400 * a * (b - a * a)
      const gb = 200 * (b - a * a)
      ;[a, b] = [a - 0.0015 * ga, b - 0.0015 * gb]
      if (i % 8 === 0) {
        px.push(a)
        py.push(b)
      }
    }
    return { x: px, y: py }
  }, [])
  const lv = useMemo(() => grid(0.5, 7, levels.value), [levels.value])
  const x = useAxis({ label: 'a' })
  const y = useAxis({ label: 'b', equal: x })
  return (
    <Figure
      title="Contours: a loss landscape and an optimiser"
      // TODO(5c): purpose taken from the description
      purpose="Level sets of log(1 + Rosenbrock), (1 − a)² + 100(b − a²)², by marching squares, with gradient-descent iterates."
      controls={<Slider label="levels" param={levels} />}
    >
      <Plot x={x} y={y}>
        <Contours x={xs} y={ys} z={z} levels={lv} labels={false} />
        <Curve name="gradient descent" x={iterates.x} y={iterates.y} showPoints slot={1} />
        <Points name="minimum" x={[1]} y={[1]} emphasis />
      </Plot>
    </Figure>
  )
}

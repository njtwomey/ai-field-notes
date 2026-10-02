/** UI-kit figures for `Plots`: shared axis models, equal units, a categorical x and a raster aligned with curves. */
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Normal } from 'aifn/probability/distributions'
import { useMemo, useState } from 'react'
import { Slider, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import {
  Bars,
  Curve,
  Density,
  formatNumber,
  Handle,
  Histogram,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  useAxis,
} from '@lab/viz'
import { density2, grid } from './data'

/** f above with equal units (a round unit circle), f′ below; one x axis model, one x₀ handle on both. */
export function SharedAxisFigure() {
  const omega = useParam(1.5, { min: 0.5, max: 4 })
  const x0 = useParam(1, { min: -4, max: 4 })
  const x = useAxis({ label: 'x' })
  const yf = useAxis({ label: 'f', equal: x })
  const yd = useAxis({ label: 'f′', hold: 'union' })
  const { xs, f, df } = useMemo(() => {
    const w = omega.value
    const xs = grid(-4, 4, 321)
    return {
      xs,
      f: xs.map((v) => Math.sin(w * v) * Math.exp(-(v * v) / 8)),
      df: xs.map((v) => (w * Math.cos(w * v) - (v / 4) * Math.sin(w * v)) * Math.exp(-(v * v) / 8)),
    }
  }, [omega.value])
  const w = omega.value
  const at = x0.value
  const f0 = Math.sin(w * at) * Math.exp(-(at * at) / 8)
  const t = useMemo(() => grid(0, 2 * Math.PI, 97), [])
  return (
    <Figure
      title="Plots: two panels sharing one x axis"
      // TODO(5c): purpose taken from the description
      purpose="Both Plots take the same x model: one range, one zoom, x labels on the lower panel only. The upper panel's y is linked to x for equal units."
      defaultSize="L"
      controls={
        <>
          <Slider label="ω" param={omega} />
          <Slider label="x₀" param={x0} />
        </>
      }
      readouts={<Readout label="f(x₀)" value={formatNumber(f0)} />}
      caption="The grid sizes the upper panel so its units are equal (the unit circle at x₀ is round) and gives the lower panel the rest. Zoom x from the toolbar and both panels move. Drag x₀ on either panel."
    >
      <Plots rows={2} heights={[2, 1]} hoverGroup>
        <Plot x={x} y={yf}>
          <Curve name="f(x)" x={xs} y={f} />
          <Curve
            name="unit circle at x₀"
            x={t.map((a) => at + Math.cos(a))}
            y={t.map((a) => f0 + Math.sin(a))}
            slot={3}
            thin
            live
          />
          <Handle kind="x" at={at} label="x₀" onDrag={x0.set} />
        </Plot>
        <Plot x={x} y={yd}>
          <Curve name="f′(x)" x={xs} y={df} slot={1} />
          <Handle kind="x" at={at} label="x₀" onDrag={x0.set} />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** A change of variables: the map, the output density rotated beside it, the input density below it. */
export function ChangeOfVariablesFigure() {
  const scale = useParam(1.5, { min: 0.3, max: 4 })
  const input = useMemo(() => Normal(0, 1), [])
  const samples = useMemo(() => toFlat(input.sample(stream('cov-kit'), { shape: [4000] }) as Tensor), [input])
  const g = (v: number) => Math.tanh(scale.value * v)
  const xs = useMemo(() => grid(-3.5, 3.5, 281), [])
  const gx = useMemo(() => xs.map(g), [xs, scale.value]) // eslint-disable-line react-hooks/exhaustive-deps -- g is a function of scale
  const outputs = useMemo(() => samples.map(g), [samples, scale.value]) // eslint-disable-line react-hooks/exhaustive-deps -- as above
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y = tanh(s·x)', range: [-1.05, 1.05] })
  const py = useAxis({ label: 'p(y)', hold: 'union' })
  const px = useAxis({ label: 'p(x)' })
  return (
    <Figure
      title="Plots: a map with its input and output densities"
      // TODO(5c): purpose taken from the description
      purpose="y = tanh(s·x) for x ~ N(0, 1): the map (top left), a histogram of y laid along y beside it (top right, sharing y), and the input below (sharing x)."
      defaultSize="L"
      controls={<Slider label="s" param={scale} />}
      caption="One y model is shared across the top row and one x model down the left column, so the panels line up edge to edge. The rotated histogram uses orient y."
    >
      <Plots rows={2} cols={2} heights={[3, 2]} widths={[3, 1.2]}>
        <Plot x={x} y={y}>
          <Curve name="g(x)" x={xs} y={gx} emphasis />
        </Plot>
        <Plot x={py} y={y}>
          <Histogram name="y draws" values={outputs} orient="y" bins={40} slot={1} />
        </Plot>
        <Plot x={x} y={px}>
          <Histogram name="x draws" values={samples} bins={40} />
          <Density dist={input} name="N(0, 1)" emphasis />
        </Plot>
        <div aria-hidden />
      </Plots>
    </Figure>
  )
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** A categorical x shared by two panels: counts per month and a rate per month. */
export function CategoricalFigure() {
  const counts = [31, 28, 35, 40, 52, 61, 70, 66, 50, 42, 33, 30]
  const rate = counts.map((c, i) => c / (20 + 3 * i))
  const idx = MONTHS.map((_, i) => i)
  const x = useAxis({ label: 'month', categories: MONTHS })
  const yc = useAxis({ label: 'count' })
  const yr = useAxis({ label: 'rate' })
  return (
    <Figure
      title="Plots: a categorical x axis in two panels"
      // TODO(5c): purpose taken from the description
      purpose="Categories at integer positions, labelled by name; a bar chart above a line through the same categories."
      caption="Hover either panel: the month and both values show in the tooltip and the readout."
    >
      <Plots rows={2} heights={[2, 1]} hoverGroup>
        <Plot x={x} y={yc}>
          <Bars name="count" x={idx} y={counts} />
        </Plot>
        <Plot x={x} y={yr}>
          <Curve name="rate" x={idx} y={rate} slot={1} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** A raster with its colour bar above a slice curve: the column keeps the bar's room, so the plot edges align. */
export function RasterAlignedFigure() {
  const [level, setLevel] = useState(0.5)
  const xs = useMemo(() => grid(-3.5, 3.5, 71), [])
  const z = useMemo(() => xs.map((yv) => xs.map((xv) => density2(xv, yv))), [xs])
  const slice = useMemo(() => xs.map((xv) => density2(xv, level)), [xs, level])
  const x = useAxis({ label: 'x' })
  const y = useAxis({ label: 'y' })
  const v = useAxis({ label: 'density at y₀', hold: 'union' })
  return (
    <Figure
      title="Plots: a raster aligned with a curve"
      // TODO(5c): purpose taken from the description
      purpose="The bump density as a raster (with its colour bar) above its slice at y = y₀. The colour bar's room is kept down the whole column."
      defaultSize="L"
      readouts={<Readout label="y₀" value={formatNumber(level)} />}
      caption="Drag y₀ on the raster (a horizontal handle). The raster's plot area and the curve's start and end at the same pixels, though only the raster has a colour bar."
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={x} y={y}>
          <Raster x={xs} y={xs} z={z} valueLabel="density" />
          <Handle kind="y" at={level} label="y₀" onDrag={(l) => setLevel(Math.max(-3.5, Math.min(3.5, l)))} />
        </Plot>
        <Plot x={x} y={v}>
          <Curve name="density at y₀" x={xs} y={slice} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** A calibration curve on the unit square, square, with a tight strip of bin counts under it. */
export function UnitSquareFigure() {
  const bins = grid(0.05, 0.95, 10)
  const observed = bins.map((b) => Math.min(1, Math.max(0, b + 0.12 * Math.sin(6 * b))))
  const counts = bins.map((b) => Math.round(400 * Math.exp(-((b - 0.3) ** 2) / 0.08) + 20))
  const x = useAxis({ label: 'predicted probability', range: [0, 1] })
  const y = useAxis({ label: 'observed frequency', range: [0, 1], equal: x })
  const n = useAxis({ label: 'count' })
  return (
    <Figure
      title="Plots: a square unit square with a tight strip"
      // TODO(5c): purpose taken from the description
      purpose="A reliability diagram on [0, 1]² with equal units (square, diagonal at 45°) and its bin counts in a tight strip sharing x."
      caption="ratiosOf equal sizes the strip from the square's plot height; tight puts the two nearly edge to edge."
    >
      <Plots rows={2} heights={[1, 0.25]} ratiosOf="equal" tight>
        <Plot x={x} y={y}>
          <Curve name="perfect calibration" x={[0, 1]} y={[0, 1]} muted dashed />
          <Curve name="observed" x={bins} y={observed} showPoints />
          <Points name="bins" x={bins} y={observed} slot={0} live />
        </Plot>
        <Plot x={x} y={n}>
          <Bars name="count" x={bins} y={counts} width={0.08} muted />
        </Plot>
      </Plots>
    </Figure>
  )
}

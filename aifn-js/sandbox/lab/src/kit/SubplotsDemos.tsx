import { grad } from 'aifn/foundation/autodiff'
import { exp, linspace, mul, sin, square, sum, toFlat, type Tensor, type Value } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Slider, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Curve, formatNumber, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

/** f(x) = sin(ωx)·e^(−x²/8), written once with aifn primitives so grad gives f′. */
const wave =
  (omega: number) =>
  (x: Value): Value =>
    mul(sin(mul(omega, x)), exp(mul(-1 / 8, square(x))))

/** Two rows sharing x: f above (equal aspect, twice the height), f′ = grad(f) below, one draggable x₀ for both. */
export function SharedXFigure() {
  const omega = useParam(1.5, { min: 0.5, max: 4 })
  const x0 = useParam(1, { min: -4, max: 4 })
  const { top, bottom, y0, slope } = useMemo(() => {
    const f = wave(omega.value)
    const x = linspace(-4, 4, 321)
    const xs = toFlat(x)
    const df = grad((u: Value) => sum(f(u)))
    const top = { x: xs, y: toFlat(f(x) as Tensor) }
    const bottom = { x: xs, y: toFlat(df(x) as Tensor) }
    return { top, bottom, y0: f(x0.value) as number, slope: grad(f)(x0.value) as number }
  }, [omega.value, x0.value])
  const xAxis = useAxis({ label: 'x' })
  const fAxis = useAxis({ label: 'f', equal: xAxis })
  const dfAxis = useAxis({ label: 'f′' })
  // A live tangent and a unit circle at x₀: with equal units the circle is round.
  const tangent = useMemo(() => {
    const t = Array.from({ length: 97 }, (_, k) => (2 * Math.PI * k) / 96)
    return {
      line: { x: [x0.value - 1, x0.value + 1], y: [y0 - slope, y0 + slope] },
      circle: { x: t.map((a) => x0.value + Math.cos(a)), y: t.map((a) => y0 + Math.sin(a)) },
    }
  }, [x0.value, y0, slope])
  return (
    <Figure
      title="Plots: two rows sharing x"
      // TODO(5c): purpose taken from the description
      purpose="f above at twice the height, f′ = grad(f) below. The plot areas share left and right edges."
      defaultSize="L"
      controls={
        <>
          <Slider label="ω" param={omega} />
          <Slider label="x₀" param={x0} />
        </>
      }
      readouts={
        <>
          <Readout label="f(x₀)" value={formatNumber(y0)} />
          <Readout label="f′(x₀)" value={formatNumber(slope)} />
        </>
      }
      caption="Zoom or pan x on either panel and both move; hover either and both show the pointer. The top panel's f axis has equal units with x (useAxis equal): the grid sets its height from f's fitted ranges so one unit is as long on y as on x (the unit circle is round), and the bottom panel takes the rest of the frame. A very tall panel would narrow the column instead of stretching past the page. Drag x₀ in either panel."
    >
      <Plots rows={2} heights={[2, 1]} hoverGroup>
        <Plot x={xAxis} y={fAxis}>
          <Curve name="f(x)" x={top.x} y={top.y} slot={0} />
          <Curve id="tangent" name="tangent" x={tangent.line.x} y={tangent.line.y} slot={2} dashed live />
          <Curve id="circle" name="unit circle at x₀" x={tangent.circle.x} y={tangent.circle.y} slot={3} thin live />
          <Handle kind="x" at={x0.value} label="x₀" onDrag={x0.set} />
        </Plot>
        <Plot x={xAxis} y={dfAxis}>
          <Curve name="f′(x)" x={bottom.x} y={bottom.y} slot={1} />
          <Handle kind="x" at={x0.value} label="x₀" onDrag={x0.set} />
        </Plot>
      </Plots>
    </Figure>
  )
}

/** Two columns sharing y: the same densities against x and against log x. */
export function SharedYFigure() {
  const series = useMemo(() => {
    const x = grid(-4, 4, 201)
    const normal = x.map((v) => Math.exp((-v * v) / 2) / Math.sqrt(2 * Math.PI))
    const laplace = x.map((v) => Math.exp(-Math.abs(v) * Math.SQRT2) / Math.SQRT2)
    const t = grid(0.01, 6, 300)
    const gamma = t.map((v) => v * Math.exp(-v))
    const exponential = t.map((v) => Math.exp(-v))
    return {
      left: [
        { name: 'normal', x, y: normal, slot: 0 },
        { name: 'Laplace', x, y: laplace, slot: 1 },
      ],
      right: [
        { name: 'gamma(2, 1)', x: t, y: gamma, slot: 2 },
        { name: 'exponential(1)', x: t, y: exponential, slot: 3 },
      ],
    }
  }, [])
  const xAxis = useAxis({ label: 'x' })
  const tAxis = useAxis({ label: 't' })
  const density = useAxis({ label: 'density' })
  return (
    <Figure
      title="Plots: two columns sharing y"
      // TODO(5c): purpose taken from the description
      purpose="Densities on the real line (left) and on the positive half-line (right), on one density scale."
      caption="Only the left panel labels y; the plot areas share top and bottom edges. Zoom y on either and both follow."
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={xAxis} y={density}>
          {series.left.map((c) => (
            <Curve key={c.name} {...c} />
          ))}
        </Plot>
        <Plot x={tAxis} y={density}>
          {series.right.map((c) => (
            <Curve key={c.name} {...c} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

/** A 2 × 2 grid: columns share x, rows share y. */
export function GridFigure() {
  const cells = useMemo(() => {
    const a = grid(0, 2 * Math.PI, 200)
    const b = grid(0.1, 10, 200)
    return [
      { name: 'sin', x: a, y: a.map(Math.sin), slot: 0 },
      { name: 'log', x: b, y: b.map((v) => Math.log(v) / 2.5), slot: 1 },
      { name: 'sin²', x: a, y: a.map((v) => 3 * Math.sin(v) ** 2), slot: 2 },
      { name: '1000/x', x: b, y: b.map((v) => 1000 / (v * v + 400)), slot: 3 },
    ]
  }, [])
  const theta = useAxis({ label: 'θ' })
  const x = useAxis({ label: 'x (0.1 – 10)' })
  const top = useAxis({ label: 'value' })
  const bottom = useAxis({ label: 'value' })
  return (
    <Figure
      title="Plots: a 2 × 2 grid"
      // TODO(5c): purpose taken from the description
      purpose="Columns share an x axis model, rows a y axis model; hover is linked within the grid."
      defaultSize="L"
      caption="Every column has its own x range and every row its own y range. Tick labels appear only on the bottom row and the left column. The bottom-right panel's y labels differ in width from the bottom-left's, but plot edges still line up."
    >
      <Plots rows={2} cols={2} hoverGroup>
        {cells.map((c, i) => (
          <Plot key={i} x={i % 2 ? x : theta} y={i < 2 ? top : bottom} legend={false}>
            <Curve {...c} />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}

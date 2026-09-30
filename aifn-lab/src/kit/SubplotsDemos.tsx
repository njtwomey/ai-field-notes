import { grad } from 'aifn/autodiff'
import { exp, linspace, mul, sin, square, sum, toFlat, type Tensor, type Value } from 'aifn/tensor'
import { useMemo } from 'react'
import { Slider, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { formatNumber, Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'

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
    const top: XYSeries[] = [{ name: 'f(x)', type: 'line', x: xs, y: toFlat(f(x) as Tensor), slot: 0 }]
    const bottom: XYSeries[] = [{ name: 'f′(x)', type: 'line', x: xs, y: toFlat(df(x) as Tensor), slot: 1 }]
    return { top, bottom, y0: f(x0.value) as number, slope: grad(f)(x0.value) as number }
  }, [omega.value, x0.value])
  const handles = useMemo(
    (): Handle[] => [{ kind: 'x', at: x0.value, label: 'x₀', onDrag: x0.set }],
    [x0.value, x0.set],
  )
  // A live tangent and a unit circle at x₀: with equal units the circle is round.
  const tangent = useMemo((): XYSeries[] => {
    const t = Array.from({ length: 97 }, (_, k) => (2 * Math.PI * k) / 96)
    return [
      {
        name: 'tangent',
        type: 'line',
        x: [x0.value - 1, x0.value + 1],
        y: [y0 - slope, y0 + slope],
        slot: 2,
        dashed: true,
      },
      {
        name: 'unit circle at x₀',
        type: 'line',
        x: t.map((a) => x0.value + Math.cos(a)),
        y: t.map((a) => y0 + Math.sin(a)),
        slot: 3,
        thin: true,
      },
    ]
  }, [x0.value, y0, slope])
  return (
    <Figure
      title="Subplots: two rows sharing x"
      description="f above at twice the height, f′ = grad(f) below. The plot areas share left and right edges."
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
      caption="Zoom or pan x on either panel and both move; hover either and both show the pointer. The top panel asks for equal units (Panel aspect equal): the grid sets its height from f's fitted ranges so one unit is as long on y as on x (the unit circle is round), and the bottom panel takes the rest of the frame. A very tall panel would narrow the column instead of stretching past the page. Drag x₀ in either panel."
    >
      <Subplots rows={2} sharex heightRatios={[2, 1]} hoverGroup>
        <Panel aspect="equal">
          <XYChart series={top} live={tangent} xLabel="x" yLabel="f" handles={handles} />
        </Panel>
        <Panel>
          <XYChart series={bottom} xLabel="x" yLabel="f′" handles={handles} />
        </Panel>
      </Subplots>
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
        { name: 'normal', type: 'line', x, y: normal, slot: 0 },
        { name: 'Laplace', type: 'line', x, y: laplace, slot: 1 },
      ] satisfies XYSeries[],
      right: [
        { name: 'gamma(2, 1)', type: 'line', x: t, y: gamma, slot: 2 },
        { name: 'exponential(1)', type: 'line', x: t, y: exponential, slot: 3 },
      ] satisfies XYSeries[],
    }
  }, [])
  return (
    <Figure
      title="Subplots: two columns sharing y"
      description="Densities on the real line (left) and on the positive half-line (right), on one density scale."
      caption="Only the left panel labels y; the plot areas share top and bottom edges. Zoom y on either and both follow."
    >
      <Subplots cols={2} sharey widthRatios={[3, 2]}>
        <Panel>
          <XYChart series={series.left} xLabel="x" yLabel="density" />
        </Panel>
        <Panel>
          <XYChart series={series.right} xLabel="t" yLabel="density" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

/** A 2 × 2 grid: columns share x, rows share y. */
export function GridFigure() {
  const cells = useMemo(() => {
    const a = grid(0, 2 * Math.PI, 200)
    const b = grid(0.1, 10, 200)
    return [
      [{ name: 'sin', type: 'line', x: a, y: a.map(Math.sin), slot: 0 }],
      [{ name: 'log', type: 'line', x: b, y: b.map((v) => Math.log(v) / 2.5), slot: 1 }],
      [{ name: 'sin²', type: 'line', x: a, y: a.map((v) => 3 * Math.sin(v) ** 2), slot: 2 }],
      [{ name: '1000/x', type: 'line', x: b, y: b.map((v) => 1000 / (v * v + 400)), slot: 3 }],
    ] satisfies XYSeries[][]
  }, [])
  return (
    <Figure
      title="Subplots: a 2 × 2 grid"
      description="Columns share x (sharex 'col'), rows share y (sharey 'row'); hover is linked within the grid."
      defaultSize="L"
      caption="Every column has its own x range and every row its own y range. Tick labels appear only on the bottom row and the left column. The bottom-right panel's y labels differ in width from the bottom-left's, but plot edges still line up."
    >
      <Subplots rows={2} cols={2} sharex="col" sharey="row" hoverGroup>
        {cells.map((series, i) => (
          <Panel key={i}>
            <XYChart series={series} xLabel={i % 2 ? 'x (0.1 – 10)' : 'θ'} yLabel="value" />
          </Panel>
        ))}
      </Subplots>
    </Figure>
  )
}

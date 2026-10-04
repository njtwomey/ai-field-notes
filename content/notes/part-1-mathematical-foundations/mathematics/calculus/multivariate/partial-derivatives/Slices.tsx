import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Vec = [number, number]
type Fn = {
  f: (x: number, y: number) => number
  fx: (x: number, y: number) => number
  fy: (x: number, y: number) => number
}

const WIDTH = 0.5
const hill = (dx: number, dy: number) => Math.exp(-(dx * dx + dy * dy) / (2 * WIDTH ** 2))

const FUNCTIONS: Record<'hills' | 'ridge', Fn> = {
  // Two narrow bumps on a flat plain: the slices cut through clearly shaped peaks.
  hills: {
    f: (x, y) => hill(x - 1, y) + 0.8 * hill(x + 1, y - 1),
    fx: (x, y) => (-(x - 1) * hill(x - 1, y) - 0.8 * (x + 1) * hill(x + 1, y - 1)) / WIDTH ** 2,
    fy: (x, y) => (-y * hill(x - 1, y) - 0.8 * (y - 1) * hill(x + 1, y - 1)) / WIDTH ** 2,
  },
  // A diagonal ridge: steep across it, level along it, so the two partials differ sharply.
  ridge: {
    f: (x, y) => Math.exp(-2 * (x - y) ** 2) * Math.exp(-((x + y) ** 2) / 16),
    fx: (x, y) => Math.exp(-2 * (x - y) ** 2 - (x + y) ** 2 / 16) * (-4 * (x - y) - (x + y) / 8),
    fy: (x, y) => Math.exp(-2 * (x - y) ** 2 - (x + y) ** 2 / 16) * (4 * (x - y) - (x + y) / 8),
  },
}
const AXIS = toFlat(linspace(-2.5, 2.5, 51))
const LINE = toFlat(linspace(-2.5, 2.5, 200))
const TANGENT = 0.7

/** A surface with two perpendicular slices through a point; each slice's slope there is one partial derivative. */
export function Slices() {
  const state = useFigureState({
    name: choice<keyof typeof FUNCTIONS>(
      [
        { value: 'hills', label: 'two hills' },
        { value: 'ridge', label: 'ridge' },
      ],
      'hills',
      { label: 'function' },
    ),
  })
  const [point, setPoint] = useState<Vec>([0.6, 0.4])
  const fn = FUNCTIONS[state.name]
  const [px, py] = point

  const z = useMemo(() => AXIS.map((y) => AXIS.map((x) => fn.f(x, y))), [fn])
  const fx = fn.fx(px, py)
  const fy = fn.fy(px, py)
  const f0 = fn.f(px, py)

  const overlay = useMemo(
    () =>
      [
        { name: 'slice y = y₀', x: [LINE[0], LINE[LINE.length - 1]], y: [py, py], slot: 1 },
        { name: 'slice x = x₀', x: [px, px], y: [LINE[0], LINE[LINE.length - 1]], slot: 2 },
      ] as const,
    [px, py],
  )
  const slice = (along: 'x' | 'y'): SeriesSpec[] => {
    const values = LINE.map((t) => (along === 'x' ? fn.f(t, py) : fn.f(px, t)))
    const at = along === 'x' ? px : py
    const slope = along === 'x' ? fx : fy
    return [
      { name: along === 'x' ? 'f(x, y₀)' : 'f(x₀, y)', type: 'line', x: LINE, y: values, slot: along === 'x' ? 1 : 2 },
      {
        name: 'tangent',
        type: 'line',
        x: [at - TANGENT, at + TANGENT],
        y: [f0 - TANGENT * slope, f0 + TANGENT * slope],
        slot: 0,
        dashed: true,
      },
      { name: 'point', type: 'scatter', x: [at], y: [f0], emphasis: true },
    ]
  }

  const clamp = (v: number) => Math.round(Math.min(Math.max(v, AXIS[0]), AXIS[AXIS.length - 1]) * 100) / 100
  const yRange: [number, number] = [-0.1, 1.15]

  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'y' })
  const xAxis2 = useAxis({ label: 'x', hold: 'union' })
  const yAxis2 = useAxis({ label: 'f(x, y₀)', range: yRange })
  const xAxis3 = useAxis({ label: 'y', hold: 'union' })
  const yAxis3 = useAxis({ label: 'f(x₀, y)', range: yRange })
  return (
    <Figure
      title="Partial derivatives are slopes of slices"
      state={state}
      caption="Drag the point on the surface. The orange line holds y fixed and the green line holds x fixed; the two charts below show the surface along each line. The slope of each slice at the point, drawn dashed, is one partial derivative. On the ridge the slice across it is steep while the slice along it is nearly level."

      readouts={
        <>
          <Readout label="(x₀, y₀)" value={`(${formatNumber(px)}, ${formatNumber(py)})`} />
          <Readout label="∂f/∂x" value={formatNumber(fx)} />
          <Readout label="∂f/∂y" value={formatNumber(fy)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Plot x={xAxis} y={yAxis} height={400}>
          <Raster x={AXIS} y={AXIS} z={z} valueLabel={'f'} />
          <Curve {...overlay[0]} live />
          <Curve {...overlay[1]} live />
          <Handle kind="point" at={point} label="point" onDrag={([x, y]) => setPoint([clamp(x), clamp(y)])} />
        </Plot>
        <div className="space-y-2">
          <Plot x={xAxis2} y={yAxis2} height={190}>
            {seriesLayers(slice('x'))}
          </Plot>
          <Plot x={xAxis3} y={yAxis3} height={190}>
            {seriesLayers(slice('y'))}
          </Plot>
        </div>
      </div>
    </Figure>
  )
}

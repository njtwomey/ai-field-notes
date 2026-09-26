import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

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
const AXIS = linspace(-2.5, 2.5, 51)
const LINE = linspace(-2.5, 2.5, 200)
const TANGENT = 0.7

/** A surface with two perpendicular slices through a point; each slice's slope there is one partial derivative. */
export function Slices() {
  const [name, setName] = useState<keyof typeof FUNCTIONS>('hills')
  const [point, setPoint] = useState<Vec>([0.6, 0.4])
  const fn = FUNCTIONS[name]
  const [px, py] = point

  const z = useMemo(() => AXIS.map((y) => AXIS.map((x) => fn.f(x, y))), [fn])
  const fx = fn.fx(px, py)
  const fy = fn.fy(px, py)
  const f0 = fn.f(px, py)

  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      { name: 'slice y = y₀', type: 'line', x: [LINE[0], LINE[LINE.length - 1]], y: [py, py], slot: 1 },
      { name: 'slice x = x₀', type: 'line', x: [px, px], y: [LINE[0], LINE[LINE.length - 1]], slot: 2 },
    ],
    [px, py],
  )
  const slice = (along: 'x' | 'y'): XYSeries[] => {
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
  const handles: Handle[] = [
    { kind: 'point', at: point, label: 'point', onDrag: ([x, y]) => setPoint([clamp(x), clamp(y)]) },
  ]
  const yRange: [number, number] = [-0.1, 1.15]

  return (
    <Interactive
      title="Partial derivatives are slopes of slices"
      caption="Drag the point on the surface. The orange line holds y fixed and the green line holds x fixed; the two charts below show the surface along each line. The slope of each slice at the point, drawn dashed, is one partial derivative. On the ridge the slice across it is steep while the slice along it is nearly level."
      controls={
        <ParamChoice
          label="function"
          value={name}
          onChange={setName}
          options={[
            { value: 'hills', label: 'two hills' },
            { value: 'ridge', label: 'ridge' },
          ]}
        />
      }
      readout={
        <>
          <Readout label="(x₀, y₀)" value={`(${formatNumber(px)}, ${formatNumber(py)})`} />
          <Readout label="∂f/∂x" value={formatNumber(fx)} />
          <Readout label="∂f/∂y" value={formatNumber(fy)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={z}
          xLabel="x"
          yLabel="y"
          valueLabel="f"
          overlay={overlay}
          handles={handles}
          height={400}
        />
        <div className="space-y-2">
          <XYChart series={slice('x')} xLabel="x" yLabel="f(x, y₀)" yRange={yRange} height={190} />
          <XYChart series={slice('y')} xLabel="y" yLabel="f(x₀, y)" yRange={yRange} height={190} />
        </div>
      </div>
    </Interactive>
  )
}

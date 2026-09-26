import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from '@/components/viz'
import { linspace } from '@/lib/math'

type Vec = [number, number]
type Fn = { f: (x: number, y: number) => number; grad: (x: number, y: number) => Vec }

const FUNCTIONS: Record<'bowl' | 'saddle' | 'hills', Fn> = {
  bowl: { f: (x, y) => x * x + 3 * y * y, grad: (x, y) => [2 * x, 6 * y] },
  saddle: { f: (x, y) => x * x - y * y, grad: (x, y) => [2 * x, -2 * y] },
  hills: {
    f: (x, y) => Math.exp(-((x - 1) ** 2 + y * y)) + 0.8 * Math.exp(-((x + 1) ** 2 + (y - 1) ** 2)),
    grad: (x, y) => {
      const a = Math.exp(-((x - 1) ** 2 + y * y))
      const b = 0.8 * Math.exp(-((x + 1) ** 2 + (y - 1) ** 2))
      return [-2 * (x - 1) * a - 2 * (x + 1) * b, -2 * y * a - 2 * (y - 1) * b]
    },
  },
}
const AXIS = linspace(-2.5, 2.5, 51)
const ARROW = 0.9

/** A surface, the gradient at a dragged point, and the rate of change along a chosen direction. */
export function GradientField() {
  const [name, setName] = useState<keyof typeof FUNCTIONS>('bowl')
  const [point, setPoint] = useState<Vec>([1.2, 0.6])
  const direction = useParam(45, { min: 0, max: 360, step: 1 })
  const angle = direction.value
  const fn = FUNCTIONS[name]

  const z = useMemo(() => AXIS.map((y) => AXIS.map((x) => fn.f(x, y))), [fn])
  const g = fn.grad(...point)
  const norm = Math.hypot(...g)
  const u = useMemo((): Vec => [Math.cos((angle * Math.PI) / 180), Math.sin((angle * Math.PI) / 180)], [angle])
  const directional = g[0] * u[0] + g[1] * u[1]

  // Arrow of fixed length along ∇f, so direction is readable at any steepness; the size is in the readout.
  const vectors = useMemo(
    () =>
      norm > 1e-9
        ? [{ from: point, to: [point[0] + (ARROW * g[0]) / norm, point[1] + (ARROW * g[1]) / norm] as Vec }]
        : [],
    [point, g, norm],
  )
  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      {
        name: 'direction u',
        type: 'line',
        x: [point[0], point[0] + ARROW * u[0]],
        y: [point[1], point[1] + ARROW * u[1]],
        slot: 1,
      },
    ],
    [point, u],
  )

  const clamp = (v: number) => Math.round(Math.min(Math.max(v, AXIS[0]), AXIS[AXIS.length - 1]) * 100) / 100
  const handles: Handle[] = [
    { kind: 'point', at: point, label: 'point', onDrag: ([x, y]) => setPoint([clamp(x), clamp(y)]) },
    {
      kind: 'point',
      at: [point[0] + ARROW * u[0], point[1] + ARROW * u[1]],
      label: 'u',
      onDrag: ([x, y]) =>
        direction.set(((((Math.atan2(y - point[1], x - point[0]) * 180) / Math.PI) % 360) + 360) % 360),
    },
  ]

  return (
    <Interactive
      title="The gradient points uphill"
      caption="Drag the round handle to choose a point. The arrow shows the direction of ∇f; its length in the readout is the steepest rate of increase. The orange segment is a direction u; drag its end, or use the slider, to turn it. The rate of change along u is ∇f · u, largest when u lines up with the arrow and zero when u runs along the contour."
      controls={
        <>
          <ParamChoice
            label="function"
            value={name}
            onChange={setName}
            options={[
              { value: 'bowl', label: 'bowl x² + 3y²' },
              { value: 'saddle', label: 'saddle x² − y²' },
              { value: 'hills', label: 'two hills' },
            ]}
          />
          <ParamSlider label="direction u (degrees)" param={direction} />
        </>
      }
      readout={
        <>
          <Readout label="∇f" value={`(${formatNumber(g[0])}, ${formatNumber(g[1])})`} />
          <Readout label="‖∇f‖" value={formatNumber(norm)} />
          <Readout label="∇f · u" value={formatNumber(directional)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-2xl">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={z}
          xLabel="x"
          yLabel="y"
          valueLabel="f"
          vectors={vectors}
          overlay={overlay}
          handles={handles}
          height={420}
        />
      </div>
    </Interactive>
  )
}

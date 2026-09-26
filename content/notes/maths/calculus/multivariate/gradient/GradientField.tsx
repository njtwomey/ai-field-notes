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

const HILL = 0.5
const SADDLE = 1.5
const hill = (dx: number, dy: number) => Math.exp(-(dx * dx + dy * dy) / (2 * HILL ** 2))

const FUNCTIONS: Record<'bowl' | 'saddle' | 'hills', Fn> = {
  // A steep-walled basin: one minimum with elliptical contours, levelling off to a plain.
  bowl: {
    f: (x, y) => 1 - Math.exp(-(x * x + 3 * y * y) / 2),
    grad: (x, y) => {
      const e = Math.exp(-(x * x + 3 * y * y) / 2)
      return [x * e, 3 * y * e]
    },
  },
  // Rises along x and falls along y from a flat point at the origin, fading to a level plain.
  saddle: {
    f: (x, y) => (x * x - y * y) * Math.exp(-(x * x + y * y) / SADDLE),
    grad: (x, y) => {
      const e = Math.exp(-(x * x + y * y) / SADDLE)
      const q = x * x - y * y
      return [e * (2 * x - (2 * x * q) / SADDLE), e * (-2 * y - (2 * y * q) / SADDLE)]
    },
  },
  // Narrow bumps (width HILL) on a flat plain, so the peaks stand out and the plain between them is level.
  hills: {
    f: (x, y) => hill(x - 1, y) + 0.8 * hill(x + 1, y - 1),
    grad: (x, y) => {
      const a = hill(x - 1, y)
      const b = 0.8 * hill(x + 1, y - 1)
      const k = 1 / HILL ** 2
      return [-k * ((x - 1) * a + (x + 1) * b), -k * (y * a + (y - 1) * b)]
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
              { value: 'bowl', label: 'bowl' },
              { value: 'saddle', label: 'saddle' },
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

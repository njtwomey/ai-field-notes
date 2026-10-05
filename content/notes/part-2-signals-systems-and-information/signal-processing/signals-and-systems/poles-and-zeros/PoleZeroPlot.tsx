import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
} from 'aifn-render'
import { freqz, lfilter, magnitude } from 'aifn-compute/signal'
import { transferFunction } from 'aifn-compute/systems'
import { linspace, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

type Point = [number, number]
const CIRCLE = toFlat(linspace(0, 2 * Math.PI, 181))
const IMPULSE = Array.from({ length: 48 }, (_, n) => (n === 0 ? 1 : 0))
const LIMIT = 1.35
/** Decibels, floored at −80 dB. */
const db = (m: number) => Math.max(-80, 20 * Math.log10(Math.max(m, 1e-300)))
const clampPoint = ([x, y]: Point): Point => [
  Math.max(-LIMIT, Math.min(LIMIT, x)),
  // Keep the dragged member of each pair in the upper half-plane; its conjugate mirrors it below.
  Math.max(0, Math.min(LIMIT, y)),
]
/** Second-order polynomial with roots c and its conjugate, in powers of z⁻¹: 1 − 2 Re(c) z⁻¹ + |c|² z⁻². */
const pair = ([x, y]: Point) => [1, -2 * x, x * x + y * y]

/**
 * A second-order system with one conjugate pair of poles and one of zeros. Drag the poles and zeros in the z-plane;
 * the magnitude response and impulse response follow.
 */
export function PoleZeroPlot() {
  const [pole, setPole] = useState<Point>([0.9 * Math.cos(Math.PI / 4), 0.9 * Math.sin(Math.PI / 4)])
  const [zero, setZero] = useState<Point>([-1, 0])

  const r = useMemo(() => {
    const b = pair(zero)
    const a = pair(pole)
    const response = freqz(transferFunction(b, a, { dt: 1 }), { n: 512, includeNyquist: true })
    const omega = toFlat(response.f)
    const mag = toFlat(magnitude(response))
    const peak = Math.max(...mag)
    const impulse = toFlat(lfilter({ b, a }, IMPULSE).y as Tensor)
    return { omega, dbs: mag.map((m) => db(m / peak)), impulse }
  }, [pole, zero])

  const radius = Math.hypot(...pole)
  const angle = Math.atan2(pole[1], pole[0])
  const stable = radius < 1

  const plane = [
    {
      name: 'unit circle',
      x: CIRCLE.map(Math.cos),
      y: CIRCLE.map(Math.sin),
      muted: true,
    },
    { name: 'poles', x: [pole[0], pole[0]], y: [pole[1], -pole[1]], slot: 1 },
    { name: 'zeros', x: [zero[0], zero[0]], y: [zero[1], -zero[1]], slot: 0 },
  ] as const
  const response = [
    { name: '|H(e^{iω})| (dB, peak = 0)', x: r.omega.map((w) => w / Math.PI), y: r.dbs, slot: 0 },
  ] as const
  const impulse: SeriesSpec[] = [
    {
      name: 'h[n]',
      type: 'bar',
      x: r.impulse.map((_, n) => n),
      y: r.impulse.map((v) => Math.max(-50, Math.min(50, v))),
      slot: 0,
    },
  ]

  const xAxis = useAxis({ label: 'Re z', range: [-1.4, 1.4] })
  const yAxis = useAxis({ label: 'Im z', range: [-1.4, 1.4], equal: xAxis })
  const xAxis2 = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'dB', range: [-80, 5] })
  const xAxis3 = useAxis({ label: 'n', hold: 'union' })
  const yAxis3 = useAxis({ label: 'h[n]', hold: 'union' })
  return (
    <Figure
      title="Poles and zeros in the z-plane"
      caption="Drag the pole (and its mirrored conjugate) or the zero. The magnitude response at frequency ω is the product of distances from e^{iω} on the unit circle to the zeros, divided by the product of distances to the poles: a pole near the circle raises a sharp peak near its angle, a zero on the circle forces a null. Pull the pole outside the unit circle and the impulse response grows without bound."
      readouts={
        <>
          <Readout label="pole radius r" value={formatNumber(radius)} />
          <Readout label="pole angle θ/π" value={formatNumber(angle / Math.PI)} />
          <Readout label="zero radius" value={formatNumber(Math.hypot(...zero))} />
          <Readout label="3 dB bandwidth ≈ 2(1 − r)" value={stable ? `${formatNumber(2 * (1 - radius))} rad` : '—'} />
          <Readout label="causal system" value={stable ? 'stable' : 'unstable'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="mx-auto w-full max-w-sm">
          <Plot x={xAxis} y={yAxis}>
            <Curve {...plane[0]} />
            <Points {...plane[1]} />
            <Points {...plane[2]} />
            <Handle kind="point" at={pole} label="pole" onDrag={(p) => setPole(clampPoint(p))} />
            <Handle kind="point" at={zero} label="zero" onDrag={(p) => setZero(clampPoint(p))} />
          </Plot>
        </div>
        <div className="space-y-4">
          <Plot x={xAxis2} y={yAxis2} height={200}>
            <Curve {...response[0]} />
          </Plot>
          <Plot x={xAxis3} y={yAxis3} height={180}>
            {seriesLayers(impulse)}
          </Plot>
        </div>
      </div>
    </Figure>
  )
}

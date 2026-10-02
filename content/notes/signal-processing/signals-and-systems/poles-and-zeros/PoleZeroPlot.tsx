import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'
import { db, freqz, lfilter } from '@/lib/dsp'
import { linspace } from '@/lib/math'

type Point = [number, number]
const CIRCLE = linspace(0, 2 * Math.PI, 181)
const IMPULSE = Array.from({ length: 48 }, (_, n) => (n === 0 ? 1 : 0))
const LIMIT = 1.35
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
    const { omega, magnitude } = freqz(b, a, 512)
    const peak = Math.max(...magnitude)
    const impulse = Array.from(lfilter(b, a, IMPULSE))
    return { omega, dbs: magnitude.map((m) => db(m / peak, -80)), impulse }
  }, [pole, zero])

  const radius = Math.hypot(...pole)
  const angle = Math.atan2(pole[1], pole[0])
  const stable = radius < 1

  const plane: XYSeries[] = [
    {
      name: 'unit circle',
      type: 'line',
      x: CIRCLE.map(Math.cos),
      y: CIRCLE.map(Math.sin),
      muted: true,
    },
    { name: 'poles', type: 'scatter', x: [pole[0], pole[0]], y: [pole[1], -pole[1]], slot: 1 },
    { name: 'zeros', type: 'scatter', x: [zero[0], zero[0]], y: [zero[1], -zero[1]], slot: 0 },
  ]
  const handles: Handle[] = [
    { kind: 'point', at: pole, label: 'pole', onDrag: (p) => setPole(clampPoint(p)) },
    { kind: 'point', at: zero, label: 'zero', onDrag: (p) => setZero(clampPoint(p)) },
  ]
  const response: XYSeries[] = [
    { name: '|H(e^{iω})| (dB, peak = 0)', type: 'line', x: r.omega.map((w) => w / Math.PI), y: r.dbs, slot: 0 },
  ]
  const impulse: XYSeries[] = [
    {
      name: 'h[n]',
      type: 'bar',
      x: r.impulse.map((_, n) => n),
      y: r.impulse.map((v) => Math.max(-50, Math.min(50, v))),
      slot: 0,
    },
  ]

  return (
    <Interactive
      title="Poles and zeros in the z-plane"
      caption="Drag the pole (and its mirrored conjugate) or the zero. The magnitude response at frequency ω is the product of distances from e^{iω} on the unit circle to the zeros, divided by the product of distances to the poles: a pole near the circle raises a sharp peak near its angle, a zero on the circle forces a null. Pull the pole outside the unit circle and the impulse response grows without bound."
      readout={
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
          <XYChart
            series={plane}
            xLabel="Re z"
            yLabel="Im z"
            xRange={[-1.4, 1.4]}
            yRange={[-1.4, 1.4]}
            equalAspect
            handles={handles}
          />
        </div>
        <div className="space-y-4">
          <XYChart series={response} xLabel="ω / π" yLabel="dB" xRange={[0, 1]} yRange={[-80, 5]} height={200} />
          <XYChart series={impulse} xLabel="n" yLabel="h[n]" height={180} />
        </div>
      </div>
    </Interactive>
  )
}

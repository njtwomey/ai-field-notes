import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

type Vec = [number, number]
const R = 4
const clamp = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 20) / 20

/** The dot product of two draggable vectors, with the angle between them and the shadow of y on the line of x. */
export function AngleExplorer() {
  const [x, setX] = useState<Vec>([3, 1])
  const [y, setY] = useState<Vec>([1.5, 2.5])

  const r = useMemo(() => {
    const dot = x[0] * y[0] + x[1] * y[1]
    const nx = Math.hypot(...x)
    const ny = Math.hypot(...y)
    const ok = nx > 1e-9 && ny > 1e-9
    const cos = ok ? dot / (nx * ny) : NaN
    const theta = ok ? Math.acos(Math.min(1, Math.max(-1, cos))) : NaN
    const series: XYSeries[] = []
    if (ok) {
      // Foot of the perpendicular from y onto the line through x.
      const foot: Vec = [(dot / (nx * nx)) * x[0], (dot / (nx * nx)) * x[1]]
      series.push(
        {
          name: 'line through x',
          type: 'line',
          x: [(-2 * R * x[0]) / nx, (2 * R * x[0]) / nx],
          y: [(-2 * R * x[1]) / nx, (2 * R * x[1]) / nx],
          slot: 0,
          dashed: true,
        },
        { name: 'shadow of y, ‖y‖ cos θ', type: 'line', x: [0, foot[0]], y: [0, foot[1]], slot: 1 },
        { name: 'perpendicular', type: 'line', x: [foot[0], y[0]], y: [foot[1], y[1]], slot: 1, dashed: true },
      )
      // The angle as a small arc, turning from x towards y the short way.
      const a = Math.atan2(x[1], x[0])
      const cross = x[0] * y[1] - x[1] * y[0]
      const ts = linspace(0, theta, 40)
      const arc = Math.min(0.6, 0.35 * Math.min(nx, ny))
      series.push({
        name: 'angle θ',
        type: 'line',
        x: ts.map((t) => arc * Math.cos(a + Math.sign(cross || 1) * t)),
        y: ts.map((t) => arc * Math.sin(a + Math.sign(cross || 1) * t)),
        slot: 2,
      })
    }
    return { dot, nx, ny, cos, theta, series }
  }, [x, y])

  const handles: Handle[] = [
    { kind: 'point', at: x, label: 'x', onDrag: ([a, b]) => setX([clamp(a), clamp(b)]) },
    { kind: 'point', at: y, label: 'y', onDrag: ([a, b]) => setY([clamp(a), clamp(b)]) },
  ]

  return (
    <Interactive
      title="Alignment, length and angle"
      caption="Drag the tips of x and y. The solid segment along the dashed line is the shadow of y on the line through x; its signed length is ‖y‖ cos θ, and the dot product is that length times ‖x‖. At a right angle the shadow vanishes and the dot product is 0. Past a right angle the shadow points backwards and the dot product turns negative. Cosine similarity stays the same when either vector is stretched."
      readout={
        <>
          <Readout label="xᵀy" value={formatNumber(r.dot)} />
          <Readout label="‖x‖" value={formatNumber(r.nx)} />
          <Readout label="‖y‖" value={formatNumber(r.ny)} />
          <Readout label="cos θ" value={Number.isFinite(r.cos) ? formatNumber(r.cos) : 'undefined'} />
          <Readout
            label="θ"
            value={Number.isFinite(r.theta) ? `${formatNumber((r.theta * 180) / Math.PI)}°` : 'undefined'}
          />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.series}
          vectors={[
            { from: [0, 0], to: x },
            { from: [0, 0], to: y },
          ]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}

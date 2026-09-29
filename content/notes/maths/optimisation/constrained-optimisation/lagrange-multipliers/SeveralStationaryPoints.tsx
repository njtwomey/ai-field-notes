import { Readout, formatNumber } from '@/components/viz'
import { ConstrainedExplorer, type ConstrainedProblem } from './ConstrainedExplorer'

const DEG = Math.PI / 180
const R = Math.sqrt(3)

/** Maximise and minimise x²y on the circle x² + y² = 3, which has six stationary points. */
const PROBLEM: ConstrainedProblem = {
  f: (x, y) => x * x * y,
  gradF: (x, y) => [2 * x * y, x * x],
  gradG: (x, y) => [2 * x, 2 * y],
  curve: (deg) => [R * Math.cos(deg * DEG), R * Math.sin(deg * DEG)],
  tRange: [0, 360],
  closed: true,
  xRange: [-3, 3],
  yRange: [-3, 3],
  levels: [-3, -2, -1, -0.3, 0.3, 1, 2, 3],
  goal: 'max',
}

export function SeveralStationaryPoints() {
  return (
    <ConstrainedExplorer
      problem={PROBLEM}
      title="Six solutions of the Lagrange equations"
      caption="Left: contours of f(x, y) = x²y (grey), the circle x² + y² = 3, and the arrow of ∇f against the circle's normal. Drag the point around the circle. Right: f along the circle. The equations ∇f = λ∇g hold at six points: two global maxima (f = 2), two global minima (f = −2), and two points with f = 0 that are only local extrema. Only comparing values of f separates them."
      initialT={60}
      tStep={0.5}
      xLabel="x"
      yLabel="y"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="f"
      formatT={(v) => `${formatNumber(v)}°`}
      readout={({ stationary }) => (
        <Readout
          label="f at the stationary points"
          value={stationary
            .map((s) => `${formatNumber(Math.abs(s.value) < 1e-9 ? 0 : s.value)} (${s.kind})`)
            .join(', ')}
        />
      )}
    />
  )
}

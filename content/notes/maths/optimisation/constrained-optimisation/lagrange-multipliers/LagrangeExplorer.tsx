import { formatNumber } from '@/components/viz'
import { ConstrainedExplorer, type ConstrainedProblem } from './ConstrainedExplorer'

const DEG = Math.PI / 180

/** Minimise f(x, y) = (x − 2)² + 2(y − 1)² on the unit circle g(x, y) = x² + y² − 1 = 0. */
const PROBLEM: ConstrainedProblem = {
  f: (x, y) => (x - 2) ** 2 + 2 * (y - 1) ** 2,
  gradF: (x, y) => [2 * (x - 2), 4 * (y - 1)],
  gradG: (x, y) => [2 * x, 2 * y],
  curve: (deg) => [Math.cos(deg * DEG), Math.sin(deg * DEG)],
  tRange: [0, 360],
  closed: true,
  xRange: [-1.5, 3.5],
  yRange: [-1.5, 2.5],
  levels: [0.25, 1, 3, 6, 10, 15],
  goal: 'min',
}

const UNCONSTRAINED = { name: 'unconstrained minimum', points: [[2, 1]] as [number, number][] }

export function LagrangeExplorer() {
  return (
    <ConstrainedExplorer
      problem={PROBLEM}
      title="Gradients align at a constrained optimum"
      caption="Left: contours of f(x, y) = (x − 2)² + 2(y − 1)² (grey), the constraint x² + y² = 1, and the level set of f through the current point (dashed). The arrow is the direction of ∇f; the thin line through the point is the circle's normal, the direction of ∇g. Drag the point around the circle. Right: f along the circle against the angle θ; drag the vertical line to move the point. Where the arrow lies on the normal line, the level set touches the circle and f along the circle is flat."
      initialT={120}
      tStep={0.5}
      xLabel="x"
      yLabel="y"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="f"
      formatT={(v) => `${formatNumber(v)}°`}
      markers={UNCONSTRAINED}
    />
  )
}

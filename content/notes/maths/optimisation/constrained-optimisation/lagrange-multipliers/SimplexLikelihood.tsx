import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type Vec2,
  type XYSeries,
} from '@/components/viz'
import { contour, contours, sampleGrid } from './contours'

type Vec3 = [number, number, number]

/** The probability simplex drawn as an equilateral triangle: vertex i is the distribution with pᵢ = 1. */
const H = Math.sqrt(3) / 2
const VERTICES: Vec2[] = [
  [0, 0],
  [1, 0],
  [0.5, H],
]
const toPlane = (p: Vec3): Vec2 => [p[1] + 0.5 * p[2], H * p[2]]
function toSimplex([x, y]: Vec2): Vec3 {
  const p3 = y / H
  const p2 = x - 0.5 * p3
  return [1 - p2 - p3, p2, p3]
}
/** Keep a dragged point inside the simplex, away from the edges where log p is −∞. */
function clampToSimplex(p: Vec3): Vec3 {
  const floor = 0.01
  const q = p.map((v) => Math.max(v, floor))
  const total = q[0] + q[1] + q[2]
  return q.map((v) => v / total) as Vec3
}

const TRIANGLE = { x: [0, 1, 0.5, 0], y: [0, 0, H, 0] }
const X_RANGE: [number, number] = [-0.08, 1.08]
const Y_RANGE: [number, number] = [-0.08, H + 0.08]
/** Contours at fixed drops below the maximum log-likelihood. */
const DROPS = [0.5, 1, 2, 4, 8, 16]

export function SimplexLikelihood() {
  const n1 = useParam(6, { min: 1, max: 30, step: 1 })
  const n2 = useParam(3, { min: 1, max: 30, step: 1 })
  const n3 = useParam(1, { min: 1, max: 30, step: 1 })
  const [p, setP] = useState<Vec3>([0.3, 0.3, 0.4])
  const n: Vec3 = [n1.value, n2.value, n3.value]
  const total = n[0] + n[1] + n[2]
  const mle: Vec3 = [n[0] / total, n[1] / total, n[2] / total]
  const loglik = (q: Vec3) => n[0] * Math.log(q[0]) + n[1] * Math.log(q[1]) + n[2] * Math.log(q[2])
  const best = loglik(mle)
  const value = loglik(p)

  const [c1, c2, c3] = n
  const analysis = useMemo(() => {
    const ll = (x: number, y: number) => {
      const q = toSimplex([x, y])
      return q.some((v) => v <= 0) ? NaN : c1 * Math.log(q[0]) + c2 * Math.log(q[1]) + c3 * Math.log(q[2])
    }
    const grid = sampleGrid(ll, X_RANGE, Y_RANGE, 121)
    const top = ll(...toPlane([c1 / (c1 + c2 + c3), c2 / (c1 + c2 + c3), c3 / (c1 + c2 + c3)]))
    return {
      grid,
      background: contours(
        grid,
        DROPS.map((d) => top - d),
      ),
    }
  }, [c1, c2, c3])

  const series = useMemo((): XYSeries[] => {
    const level = contour(analysis.grid, value)
    const N = c1 + c2 + c3
    const [mx, my] = toPlane([c1 / N, c2 / N, c3 / N])
    return [
      { name: 'contours of ℓ(p)', type: 'line', x: analysis.background.x, y: analysis.background.y, muted: true },
      { name: 'simplex p₁ + p₂ + p₃ = 1', type: 'line', x: TRIANGLE.x, y: TRIANGLE.y, slot: 0 },
      { name: 'level set through p', type: 'line', x: level.x, y: level.y, slot: 1, dashed: true },
      { name: 'maximum n/N', type: 'scatter', x: [mx], y: [my], emphasis: true },
    ]
  }, [analysis, value, c1, c2, c3])

  // Partial derivatives nᵢ/pᵢ. The Lagrange condition says they all equal λ; their mean is the least-squares λ.
  const partials: Vec3 = [n[0] / p[0], n[1] / p[1], n[2] / p[2]]
  const lambda = (partials[0] + partials[1] + partials[2]) / 3
  const tangent: Vec3 = [partials[0] - lambda, partials[1] - lambda, partials[2] - lambda]
  const residual = Math.hypot(...tangent)
  const [px, py] = toPlane(p)
  // The tangent part of ∇ℓ, drawn in the plane of the triangle: the direction in which ℓ rises within the simplex.
  const [tx, ty] = [0, 1].map(
    (k) => tangent[0] * VERTICES[0][k] + tangent[1] * VERTICES[1][k] + tangent[2] * VERTICES[2][k],
  )
  const vectors = useMemo((): Segment[] => {
    const len = Math.hypot(tx, ty)
    return len > 1e-9 ? [{ from: [px, py], to: [px + (0.14 * tx) / len, py + (0.14 * ty) / len] }] : []
  }, [px, py, tx, ty])

  const bars = useMemo((): XYSeries[] => {
    const y = [c1 / p[0], c2 / p[1], c3 / p[2]]
    const N = c1 + c2 + c3
    return [
      { name: 'nᵢ / pᵢ', type: 'bar', x: [1, 2, 3], y, slot: 0 },
      { name: 'λ = N', type: 'line', x: [0.5, 3.5], y: [N, N], slot: 1, dashed: true },
    ]
  }, [p, c1, c2, c3])

  const handles: Handle[] = [
    { kind: 'point', at: [px, py], label: 'p', onDrag: (q) => setP(clampToSimplex(toSimplex(q))) },
  ]

  return (
    <Interactive
      title="Maximum likelihood on the probability simplex"
      caption="Left: the simplex of three-category distributions, with vertex i the distribution pᵢ = 1 (bottom left p₁, bottom right p₂, top p₃). Grey lines are contours of the log-likelihood ℓ(p) = Σ nᵢ log pᵢ, and the arrow is the part of ∇ℓ that lies along the simplex. Drag p, and set the counts with the sliders. Right: the partial derivatives nᵢ/pᵢ. At the maximum all three equal λ = N, so ∇ℓ is parallel to the normal 1 of the constraint and the arrow vanishes."
      controls={
        <>
          <ParamSlider label="count n₁" param={n1} />
          <ParamSlider label="count n₂" param={n2} />
          <ParamSlider label="count n₃" param={n3} />
        </>
      }
      readout={
        <>
          <Readout label="p" value={`(${p.map(formatNumber).join(', ')})`} />
          <Readout label="ℓ(p)" value={formatNumber(value)} />
          <Readout label="maximum n/N" value={`(${mle.map(formatNumber).join(', ')})`} />
          <Readout label="ℓ(n/N)" value={formatNumber(best)} />
          <Readout label="λ = mean of nᵢ/pᵢ" value={formatNumber(lambda)} />
          <Readout label="‖∇ℓ − λ1‖" value={formatNumber(residual)} />
          <Readout label="N" value={total} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={series}
          vectors={vectors}
          handles={handles}
          xRange={X_RANGE}
          yRange={Y_RANGE}
          equalAspect
          bare
        />
        <XYChart
          height={300}
          series={bars}
          xRange={[0, 4]}
          yRange={[0, 3 * total]}
          xLabel="category i"
          yLabel="∂ℓ/∂pᵢ"
        />
      </div>
    </Interactive>
  )
}

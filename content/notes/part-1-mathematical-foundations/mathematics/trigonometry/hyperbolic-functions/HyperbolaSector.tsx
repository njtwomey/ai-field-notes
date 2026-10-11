import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  Segments,
  Shapes,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { circle, grid, polygon, type Vec } from '../_shared/geometry'

const T_MAX = 2
const U = grid(-2.1, 2.1, 241)
const BRANCH = { x: U.map(Math.cosh), y: U.map(Math.sinh) }
const UNIT = circle(1)
const X1 = [{ from: [1, -5] as const, to: [1, 5] as const }]
const ASYMPTOTES = [
  { from: [0, 0] as const, to: [5, 5] as const },
  { from: [0, 0] as const, to: [5, -5] as const },
]

/** Area of a simple polygon by the shoelace formula. */
function shoelace(points: Vec[]): number {
  let s = 0
  for (let i = 0; i < points.length; i++) {
    const [x0, y0] = points[i]
    const [x1, y1] = points[(i + 1) % points.length]
    s += x0 * y1 - x1 * y0
  }
  return Math.abs(s) / 2
}

/**
 * The unit hyperbola x² − y² = 1 parametrised by (cosh t, sinh t). The shaded hyperbolic sector, between the x-axis,
 * the ray to the point and the hyperbola, has area t/2, as a circular sector of angle θ has area θ/2.
 */
export function HyperbolaSector() {
  const state = useFigureState({
    t: slider(-T_MAX, T_MAX, 1, { step: 0.01, label: 'parameter t' }),
  })
  const { t } = state
  const c = Math.cosh(t)
  const s = Math.sinh(t)
  const th = Math.tanh(t)

  const g = useMemo(() => {
    const arcT = grid(0, t, 120)
    const edge: Vec[] = arcT.map((u) => [Math.cosh(u), Math.sinh(u)])
    const outline: Vec[] = [[0, 0], ...edge]
    return {
      fill: [polygon(outline, 0, 0.2)],
      area: shoelace(outline),
      ray: [{ from: [0, 0] as const, to: [c, s] as const }],
      cosh: [{ from: [0, 0] as const, to: [c, 0] as const }],
      sinh: [{ from: [c, 0] as const, to: [c, s] as const }],
      tanh: [{ from: [1, 0] as const, to: [1, th] as const }],
    }
  }, [t, c, s, th])

  const x = useAxis({ label: 'x', range: [-1.2, 4] })
  const y = useAxis({ label: 'y', range: [-3.9, 3.9], equal: x })
  return (
    <Figure
      title="The hyperbolic sector"
      state={state}
      caption="Drag the point along the right branch of the unit hyperbola x² − y² = 1. The point is (cosh t, sinh t): the green segment is cosh t and the yellow segment is sinh t. The shaded region between the x-axis, the ray to the point and the hyperbola has area t/2, computed in the readouts from the shaded polygon. On the unit circle (grey) a sector of angle θ also has area θ/2, which is the sense in which t is a hyperbolic angle. The ray meets the tangent line x = 1 at height tanh t (orange); as t grows the ray approaches the asymptote y = x (dashed), so tanh t approaches 1 but never reaches it."
      readouts={
        <>
          <Readout label="t" value={formatNumber(t)} />
          <Readout label="shaded area" value={formatNumber(g.area)} />
          <Readout label="t / 2" value={formatNumber(Math.abs(t) / 2)} />
          <Readout label="cosh t" value={formatNumber(c)} />
          <Readout label="sinh t" value={formatNumber(s)} />
          <Readout label="tanh t" value={formatNumber(th)} />
          <Readout label="cosh²t − sinh²t" value={formatNumber(c * c - s * s)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Shapes name="hyperbolic sector" shapes={g.fill} />
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
          <Segments name="asymptotes y = ±x" segments={ASYMPTOTES} muted dashed />
          <Segments name="tangent line x = 1" segments={X1} muted />
          <Curve name="x² − y² = 1" x={BRANCH.x} y={BRANCH.y} emphasis silent />
          <Segments name="ray" segments={g.ray} emphasis width={1.5} />
          <Segments name="cosh t" segments={g.cosh} slot={2} width={4} />
          <Segments name="sinh t" segments={g.sinh} slot={3} width={4} />
          <Segments name="tanh t" segments={g.tanh} slot={1} width={4} />
          <Handle
            kind="point"
            at={[c, s]}
            label="(cosh t, sinh t)"
            onDrag={([, b]) => state.set('t', Math.max(-T_MAX, Math.min(T_MAX, Math.asinh(b))))}
          />
        </Plot>
      </div>
    </Figure>
  )
}

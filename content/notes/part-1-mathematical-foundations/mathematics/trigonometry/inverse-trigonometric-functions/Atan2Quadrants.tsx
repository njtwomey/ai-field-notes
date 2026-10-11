import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { arc, toDegrees } from '../_shared/geometry'

const R = 3
const LABELS = [
  { at: [2, 2] as const, text: 'I' },
  { at: [-2, 2] as const, text: 'II' },
  { at: [-2, -2] as const, text: 'III' },
  { at: [2, -2] as const, text: 'IV' },
]
const deg = (a: number) => (Number.isFinite(a) ? `${formatNumber(toDegrees(a))}°` : 'undefined')

/** The quadrant of (x, y), or the axis it lies on. */
function quadrant(x: number, y: number): string {
  if (x === 0 && y === 0) return 'origin'
  if (x === 0) return y > 0 ? 'positive y-axis' : 'negative y-axis'
  if (y === 0) return x > 0 ? 'positive x-axis' : 'negative x-axis'
  if (x > 0) return y > 0 ? 'I' : 'IV'
  return y > 0 ? 'II' : 'III'
}

/**
 * arctan(y/x) sees only the ratio, so it cannot tell (x, y) from (−x, −y) and always answers in the right half-plane.
 * atan2(y, x) uses both signs and returns the direction of the point in (−π, π].
 */
export function Atan2Quadrants() {
  const state = useFigureState({
    x: slider(-R, R, -2, { step: 0.05, onChart: true }),
    y: slider(-R, R, 1.5, { step: 0.05, onChart: true }),
  })
  const { x, y } = state

  const g = useMemo(() => {
    const a2 = Math.atan2(y, x)
    const a1 = x !== 0 ? Math.atan(y / x) : NaN
    const r = Math.hypot(x, y)
    const sweep = arc(0.6, 0, a2, 64)
    const ray = Number.isFinite(a1)
      ? [{ from: [0, 0] as const, to: [2.6 * Math.cos(a1), 2.6 * Math.sin(a1)] as const }]
      : []
    const arrow: Vector[] = [{ from: [0, 0], to: [x, y] }]
    return { a1, a2, r, sweep, ray, arrow }
  }, [x, y])

  const xa = useAxis({ label: 'x', range: [-R, R] })
  const ya = useAxis({ label: 'y', range: [-R, R], equal: xa })
  return (
    <Figure
      title="Why atan2 needs both coordinates"
      state={state}
      caption="Drag the point (x, y). The black arrow points to it and the blue arc sweeps its angle atan2(y, x), measured from the positive x-axis, anticlockwise positive, in (−180°, 180°]. The dashed orange ray is the direction arctan(y/x). In quadrants I and IV the two agree. In quadrants II and III the ratio y/x is the same as for the opposite point (−x, −y), and arctan returns the direction of that opposite point, off by 180°. On the y-axis the ratio is undefined, but atan2 returns ±90°."
      readouts={
        <>
          <Readout label="quadrant" value={quadrant(x, y)} />
          <Readout label="r = √(x² + y²)" value={formatNumber(g.r)} />
          <Readout label="y / x" value={x !== 0 ? formatNumber(y / x) : 'undefined'} />
          <Readout label="arctan(y/x)" value={deg(g.a1)} />
          <Readout label="atan2(y, x)" value={x === 0 && y === 0 ? '0° by convention' : deg(g.a2)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xa} y={ya} fitHeight legend={false}>
          {LABELS.map((q) => (
            <Annotation key={q.text} at={q.at} text={q.text} muted />
          ))}
          <Curve name="atan2(y, x)" x={g.sweep.x} y={g.sweep.y} slot={0} width={3} silent />
          <Segments name="arctan(y/x)" segments={g.ray} slot={1} dashed width={2} />
          <Vectors vectors={g.arrow} />
          <Handle {...state.handle(['x', 'y'], { label: '(x, y)' })} />
        </Plot>
      </div>
    </Figure>
  )
}

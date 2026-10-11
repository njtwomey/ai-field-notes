import { useMemo } from 'react'
import {
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
import { arc, circle, TAU, toDegrees, wrapPositive } from '../_shared/geometry'

const UNIT = circle(1)
const deg = (a: number) => `${formatNumber(toDegrees(a))}°`

/**
 * The angle-sum formulas as a picture. Rotating the axes by α gives the unit vectors u = (cos α, sin α) and
 * v = (−sin α, cos α). The point at angle α + β is cos β along u plus sin β along v; reading off its x coordinate gives
 * cos(α + β) = cos α cos β − sin α sin β.
 */
export function AngleSumRotation() {
  const state = useFigureState({
    alpha: slider(0, TAU, 0.6, { step: 0.01, label: 'α (rad)' }),
    beta: slider(0, TAU, 0.9, { step: 0.01, label: 'β (rad)' }),
  })
  const { alpha, beta } = state
  const ca = Math.cos(alpha)
  const sa = Math.sin(alpha)
  const cb = Math.cos(beta)
  const sb = Math.sin(beta)

  const g = useMemo(() => {
    const u: [number, number] = [ca, sa]
    const v: [number, number] = [-sa, ca]
    const mid: [number, number] = [cb * u[0], cb * u[1]]
    const tip: [number, number] = [mid[0] + sb * v[0], mid[1] + sb * v[1]]
    const vectors: Vector[] = [
      { from: [0, 0], to: u, slot: 0, label: 'u' },
      { from: [0, 0], to: v, slot: 0, label: 'v' },
      { from: [0, 0], to: tip, label: 'P(α + β)' },
    ]
    return {
      vectors,
      along: [{ from: [0, 0] as const, to: mid }],
      across: [{ from: mid, to: tip }],
      drop: [{ from: tip, to: [tip[0], 0] as const }],
      arcA: arc(0.25, 0, alpha, 48),
      arcB: arc(0.4, alpha, alpha + beta, 48),
      tip,
    }
  }, [alpha, beta, ca, sa, cb, sb])

  const x = useAxis({ label: 'x', range: [-1.3, 1.3] })
  const y = useAxis({ label: 'y', range: [-1.3, 1.3], equal: x })
  return (
    <Figure
      title="Angle sums from a rotated frame"
      state={state}
      caption="Drag the end of u to set α and the black arrow to set β. The blue arrows u and v are the x and y axes rotated by α. In that rotated frame the point at angle α + β has coordinates (cos β, sin β), so it is reached by going cos β along u (orange) and then sin β along v (green). Adding the x coordinates of the two legs gives cos(α + β) = cos α cos β − sin α sin β, and adding the y coordinates gives sin(α + β) = sin α cos β + cos α sin β. The readouts list each term."
      readouts={{
        angles: (
          <>
            <Readout label="α" value={deg(alpha)} />
            <Readout label="β" value={deg(beta)} />
            <Readout label="α + β" value={deg(wrapPositive(alpha + beta))} />
          </>
        ),
        'x coordinate': (
          <>
            <Readout label="cos α cos β" value={formatNumber(ca * cb)} />
            <Readout label="− sin α sin β" value={formatNumber(-sa * sb)} />
            <Readout label="cos(α + β)" value={formatNumber(Math.cos(alpha + beta))} />
          </>
        ),
        'y coordinate': (
          <>
            <Readout label="sin α cos β" value={formatNumber(sa * cb)} />
            <Readout label="cos α sin β" value={formatNumber(ca * sb)} />
            <Readout label="sin(α + β)" value={formatNumber(Math.sin(alpha + beta))} />
          </>
        ),
      }}
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
          <Curve name="α" x={g.arcA.x} y={g.arcA.y} muted width={1.5} silent />
          <Curve name="β" x={g.arcB.x} y={g.arcB.y} emphasis width={1.5} silent />
          <Vectors vectors={g.vectors} />
          <Segments name="cos β along u" segments={g.along} slot={1} width={5} />
          <Segments name="sin β along v" segments={g.across} slot={2} width={4} />
          <Segments name="cos(α + β)" segments={g.drop} muted dashed />
          <Handle
            kind="point"
            at={[ca, sa]}
            label="α"
            onDrag={([a, b]) => state.set('alpha', wrapPositive(Math.atan2(b, a)))}
          />
          <Handle
            kind="point"
            at={g.tip}
            label="β"
            onDrag={([a, b]) => state.set('beta', wrapPositive(Math.atan2(b, a) - alpha))}
          />
        </Plot>
      </div>
    </Figure>
  )
}

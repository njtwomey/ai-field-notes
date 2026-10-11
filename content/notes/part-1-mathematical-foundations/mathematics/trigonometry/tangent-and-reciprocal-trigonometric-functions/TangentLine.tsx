import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { circle, TAU, toDegrees, wrapPositive } from '../_shared/geometry'

const R = 2.6
const FAR = 50
const UNIT = circle(1)
const SLOT = { tan: 0, sec: 1, cot: 2, csc: 3 } as const
type Seg = { from: readonly [number, number]; to: readonly [number, number] }
const X1: Seg[] = [{ from: [1, -FAR], to: [1, FAR] }]
const Y1: Seg[] = [{ from: [-FAR, 1], to: [FAR, 1] }]
const fmt = (v: number) => (Number.isFinite(v) && Math.abs(v) < 1e6 ? formatNumber(v) : 'undefined')

/**
 * The six trigonometric functions as lengths. The ray from the origin through the point P at angle θ meets the
 * tangent line x = 1 at height tan θ, at distance sec θ from the origin; it meets the line y = 1 at x = cot θ, at
 * distance csc θ.
 */
export function TangentLine() {
  const state = useFigureState({
    theta: slider(0, TAU, 0.9, { step: 0.01, label: 'angle θ (rad)' }),
    cot: setting(false, { label: 'show cot θ and csc θ on the line y = 1' }),
  })
  const { theta } = state
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const tan = s / c
  const cot = c / s

  const g = useMemo(() => {
    const line: Seg[] = [{ from: [-FAR * c, -FAR * s], to: [FAR * c, FAR * s] }]
    const radius: Seg[] = [{ from: [0, 0], to: [c, s] }]
    const tanSeg: Seg[] = Math.abs(c) > 1e-6 ? [{ from: [1, 0], to: [1, tan] }] : []
    const secSeg: Seg[] = Math.abs(c) > 1e-6 ? [{ from: [0, 0], to: [1, tan] }] : []
    const cotSeg: Seg[] = Math.abs(s) > 1e-6 ? [{ from: [0, 1], to: [cot, 1] }] : []
    const cscSeg: Seg[] = Math.abs(s) > 1e-6 ? [{ from: [0, 0], to: [cot, 1] }] : []
    return { line, radius, tanSeg, secSeg, cotSeg, cscSeg }
  }, [c, s, tan, cot])

  const x = useAxis({ label: 'x', range: [-R, R] })
  const y = useAxis({ label: 'y', range: [-R, R], equal: x })
  return (
    <Figure
      title="Tangent and secant as lengths"
      state={state}
      caption="Drag the point P around the unit circle. The dashed line through the origin and P cuts the circle twice; it meets the tangent line x = 1 (grey, touching the circle at (1, 0)) at the point (1, tan θ). The blue segment along the tangent line has signed length tan θ, and the orange segment from the origin has length |sec θ|: the triangle with vertices (0, 0), (cos θ, 0) and P, scaled by 1/cos θ, is the triangle with vertices (0, 0), (1, 0) and (1, tan θ). Near θ = π/2 the line becomes vertical and never meets x = 1, so tan θ and sec θ are undefined there. Switch on the second pair to see cot θ and csc θ on the line y = 1."
      readouts={
        <>
          <Readout label="θ" value={`${formatNumber(theta)} rad = ${formatNumber(toDegrees(theta))}°`} />
          <Readout label="tan θ = sin θ / cos θ" value={fmt(tan)} />
          <Readout label="sec θ = 1 / cos θ" value={fmt(1 / c)} />
          <Readout label="cot θ = cos θ / sin θ" value={fmt(cot)} />
          <Readout label="csc θ = 1 / sin θ" value={fmt(1 / s)} />
          <Readout label="1 + tan²θ − sec²θ" value={fmt(1 + tan * tan - 1 / (c * c))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight>
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
          <Segments name="tangent line x = 1" segments={X1} muted />
          {state.cot && <Segments name="line y = 1" segments={Y1} muted dashed />}
          <Segments name="line through O and P" segments={g.line} emphasis dashed width={1} />
          <Segments name="radius" segments={g.radius} emphasis width={1.5} />
          <Segments name="sec θ" segments={g.secSeg} slot={SLOT.sec} width={4} />
          <Segments name="tan θ" segments={g.tanSeg} slot={SLOT.tan} width={4} />
          {state.cot && <Segments name="csc θ" segments={g.cscSeg} slot={SLOT.csc} width={4} />}
          {state.cot && <Segments name="cot θ" segments={g.cotSeg} slot={SLOT.cot} width={4} />}
          <Points name="where the line cuts the circle" x={[c, -c]} y={[s, -s]} muted size={6} />
          <Handle
            kind="point"
            at={[c, s]}
            label="P"
            onDrag={([a, b]) => state.set('theta', wrapPositive(Math.atan2(b, a)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}

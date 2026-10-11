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
  slider,
  useAxis,
  useElementSize,
  useFigureState,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import { arc, circle, grid, TAU, toDegrees, wrapPositive } from '../_shared/geometry'

const UNIT = circle(1)
const T = grid(0, TAU, 241)
const SIN = T.map(Math.sin)
const COS = T.map(Math.cos)
const SLOT = { cos: 0, sin: 1 } as const

/**
 * The point at angle θ on the unit circle has coordinates (cos θ, sin θ). Left: the circle, with the two coordinates
 * drawn as segments. Right: the same two numbers as functions of θ over one turn.
 */
export function UnitCircleTrace() {
  const state = useFigureState({
    theta: slider(0, TAU, 1, { step: 0.01, label: 'angle θ (rad)' }),
  })
  const { theta } = state
  const c = Math.cos(theta)
  const s = Math.sin(theta)

  const [cell, size] = useElementSize<HTMLDivElement>()
  const height = Math.max(240, Math.round(size.width))

  const angleArc = useMemo(() => arc(0.25, 0, theta, 48), [theta])
  const radius = useMemo(() => [{ from: [0, 0] as const, to: [c, s] as const }], [c, s])
  const cosSeg = useMemo(() => [{ from: [0, 0] as const, to: [c, 0] as const }], [c])
  const sinSeg = useMemo(() => [{ from: [c, 0] as const, to: [c, s] as const }], [c, s])
  // The traced part of each graph, from 0 up to θ.
  const traced = useMemo(() => {
    const k = T.findIndex((t) => t > theta)
    const end = k < 0 ? T.length : k
    const t = [...T.slice(0, end), theta]
    return { t, sin: t.map(Math.sin), cos: t.map(Math.cos) }
  }, [theta])

  const cx = useAxis({ label: 'x', range: [-1.25, 1.25] })
  const cy = useAxis({ label: 'y', range: [-1.25, 1.25], equal: cx })
  const tx = useAxis({ label: 'θ (rad)', range: [0, TAU] })
  const ty = useAxis({ label: 'value', range: [-1.25, 1.25] })

  return (
    <Figure
      title="Sine and cosine from the unit circle"
      state={state}
      caption="Drag the point around the circle, or drag the vertical line on the graph. The point at angle θ is (cos θ, sin θ): the blue segment is its x coordinate, cos θ, and the orange segment is its y coordinate, sin θ. The graph plots the two coordinates against θ over one turn; the bold part is the stretch already swept. Cosine starts at 1 and sine at 0; sine reaches 1 a quarter turn later, at θ = π/2, so the sine curve is the cosine curve shifted right by π/2. Each segment is a leg of a right triangle with hypotenuse 1, so cos²θ + sin²θ = 1 at every angle."
      readouts={
        <>
          <Readout label="θ" value={`${formatNumber(theta)} rad = ${formatNumber(toDegrees(theta))}°`} />
          <Readout label="cos θ" value={formatNumber(c)} />
          <Readout label="sin θ" value={formatNumber(s)} />
          <Readout label="cos²θ + sin²θ" value={formatNumber(c * c + s * s)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div ref={cell}>
          <Plot x={cx} y={cy} fitHeight legend={false} title="unit circle">
            <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
            <Curve name="angle θ" x={angleArc.x} y={angleArc.y} emphasis width={1} silent />
            <Segments name="radius" segments={radius} emphasis width={1.5} />
            <Segments name="cos θ" segments={cosSeg} slot={SLOT.cos} width={4} />
            <Segments name="sin θ" segments={sinSeg} slot={SLOT.sin} width={4} />
            <Handle
              kind="point"
              at={[c, s]}
              label="(cos θ, sin θ)"
              onDrag={([a, b]) => state.set('theta', wrapPositive(Math.atan2(b, a)))}
            />
          </Plot>
        </div>
        <FixedHeight height={height}>
          <Plot x={tx} y={ty} title="the coordinates as functions of θ">
            <Curve name="cos θ" x={T} y={COS} slot={SLOT.cos} width={1} silent />
            <Curve name="sin θ" x={T} y={SIN} slot={SLOT.sin} width={1} silent />
            <Curve name="cos θ" x={traced.t} y={traced.cos} slot={SLOT.cos} width={3} />
            <Curve name="sin θ" x={traced.t} y={traced.sin} slot={SLOT.sin} width={3} />
            <Points name="values at θ" x={[theta, theta]} y={[c, s]} emphasis size={8} live />
            <Handle {...state.handle('theta', { label: 'θ' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}

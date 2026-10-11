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
  Shapes,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { arc, circle, sector, TAU, toDegrees, wrapPositive } from '../_shared/geometry'

const VIEW: [number, number] = [-2.3, 2.3]
const UNIT = circle(1)
const MARKS = [1, 2, 3, 4, 5, 6]

/**
 * An angle θ cuts an arc of length rθ from a circle of radius r. The radian marks along the circle show that a full
 * turn holds a little over six radians, 2π.
 */
export function ArcLength() {
  const state = useFigureState({
    theta: slider(0, TAU, 2, { step: 0.01, label: 'angle θ (rad)' }),
    r: slider(0.5, 2, 1.5, { step: 0.05, label: 'radius r' }),
  })
  const { theta, r } = state
  const p: [number, number] = [r * Math.cos(theta), r * Math.sin(theta)]

  const ring = useMemo(() => circle(r), [r])
  const marks = useMemo(
    () => ({
      x: MARKS.map((k) => r * Math.cos(k)),
      y: MARKS.map((k) => r * Math.sin(k)),
      labels: MARKS.map((k) => `${k}`),
    }),
    [r],
  )
  const path = useMemo(() => arc(r, 0, theta, 128), [r, theta])
  const fill = useMemo(() => [sector(r, 0, theta, 0, 0.15)], [r, theta])
  const rays = useMemo(
    () => [
      { from: [0, 0] as const, to: [r, 0] as const },
      { from: [0, 0] as const, to: [r * Math.cos(theta), r * Math.sin(theta)] as const },
    ],
    [r, theta],
  )

  const x = useAxis({ label: 'x', range: VIEW })
  const y = useAxis({ label: 'y', range: VIEW, equal: x })
  return (
    <Figure
      title="An angle is an arc divided by a radius"
      state={state}
      caption="Drag the point around the circle, or change the radius. The coloured arc has length s = rθ and the shaded sector has area ½r²θ. Changing r scales the arc and the sector but leaves θ = s/r unchanged, which is why the radian measures the angle alone. The numbered ticks sit one radius of arc apart: six fit in a full turn with 0.283 left over, because a full turn is 2π ≈ 6.283 radians. The dashed circle is the unit circle, on which arc length and angle are the same number."
      readouts={
        <>
          <Readout label="θ (radians)" value={formatNumber(theta)} />
          <Readout label="θ (degrees)" value={`${formatNumber(toDegrees(theta))}°`} />
          <Readout label="θ (turns)" value={formatNumber(theta / TAU)} />
          <Readout label="arc length s = rθ" value={formatNumber(r * theta)} />
          <Readout label="sector area ½r²θ" value={formatNumber(0.5 * r * r * theta)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight>
          <Shapes name="sector ½r²θ" shapes={fill} />
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted dashed silent />
          <Curve name="circle of radius r" x={ring.x} y={ring.y} muted silent />
          <Segments name="rays" segments={rays} emphasis width={1.5} />
          <Curve name="arc s = rθ" x={path.x} y={path.y} slot={0} width={4} silent />
          <Points name="one radian apart" x={marks.x} y={marks.y} labels={marks.labels} muted size={6} />
          <Handle
            kind="point"
            at={p}
            label="θ"
            onDrag={([a, b]) => state.set('theta', Math.round(wrapPositive(Math.atan2(b, a)) * 100) / 100)}
          />
        </Plot>
      </div>
    </Figure>
  )
}

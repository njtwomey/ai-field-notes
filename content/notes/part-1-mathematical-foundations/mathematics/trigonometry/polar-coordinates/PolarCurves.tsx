import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useElementSize,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import { circle, grid, TAU, toDegrees } from '../_shared/geometry'

type Shape = 'cardioid' | 'rose' | 'spiral' | 'circle'
const STEPS = 360
const VIEW = 2.3

const SHAPES: Record<Shape, { label: string; r: (t: number) => number; turns: number; rRange: [number, number] }> = {
  cardioid: { label: 'cardioid r = 1 + cos θ', r: (t) => 1 + Math.cos(t), turns: 1, rRange: [-0.2, 2.2] },
  rose: { label: 'three-petal rose r = 2 cos 3θ', r: (t) => 2 * Math.cos(3 * t), turns: 1, rRange: [-2.2, 2.2] },
  spiral: { label: 'spiral r = θ / 2π', r: (t) => t / TAU, turns: 2, rRange: [0, 2.2] },
  circle: { label: 'circle r = 2 cos θ', r: (t) => 2 * Math.cos(t), turns: 0.5, rRange: [-2.2, 2.2] },
}

// The polar grid: circles of radius 0.5, 1, 1.5, 2 and rays every 30°.
const RINGS = [0.5, 1, 1.5, 2].map((r) => circle(r))
const RAYS = Array.from({ length: 12 }, (_, k) => {
  const a = (k * Math.PI) / 6
  return { from: [0, 0] as const, to: [2.2 * Math.cos(a), 2.2 * Math.sin(a)] as const }
})

/**
 * A polar curve r = f(θ) traced as θ sweeps. Left: the point (r cos θ, r sin θ) in the plane, with its radius arrow.
 * Right: the same function as an ordinary graph of r against θ.
 */
export function PolarCurves() {
  const state = useFigureState({
    shape: choice<Shape>(
      (Object.keys(SHAPES) as Shape[]).map((k) => ({ value: k, label: SHAPES[k].label })),
      'cardioid',
      { label: 'curve' },
    ),
  })
  const spec = SHAPES[state.shape]
  const [step, setStep] = useState(0)

  const full = useMemo(() => {
    const t = grid(0, spec.turns * TAU, STEPS + 1)
    const r = t.map(spec.r)
    return { t, r, x: r.map((ri, i) => ri * Math.cos(t[i])), y: r.map((ri, i) => ri * Math.sin(t[i])) }
  }, [spec])

  const theta = full.t[step]
  const r = full.r[step]
  const p: [number, number] = [r * Math.cos(theta), r * Math.sin(theta)]
  const traced = useMemo(
    () => ({
      x: full.x.slice(0, step + 1),
      y: full.y.slice(0, step + 1),
      t: full.t.slice(0, step + 1),
      r: full.r.slice(0, step + 1),
    }),
    [full, step],
  )
  const arrow = useMemo((): Vector[] => [{ from: [0, 0], to: [r * Math.cos(theta), r * Math.sin(theta)] }], [r, theta])
  // The ray at angle θ, drawn even when r < 0 sends the point the opposite way.
  const ray = useMemo(
    () => [{ from: [0, 0] as const, to: [2.2 * Math.cos(theta), 2.2 * Math.sin(theta)] as const }],
    [theta],
  )

  const [cell, size] = useElementSize<HTMLDivElement>()
  const height = Math.max(240, Math.round(size.width))

  const px = useAxis({ label: 'x', range: [-VIEW, VIEW] })
  const py = useAxis({ label: 'y', range: [-VIEW, VIEW], equal: px })
  const tx = useAxis({ label: 'θ (rad)', range: [0, spec.turns * TAU] })
  const ry = useAxis({ label: 'r', range: spec.rRange })

  return (
    <Figure
      title="Tracing a polar curve"
      state={state}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={STEPS + 1}
          label="θ"
          format={(k) => `${formatNumber(full.t[k])} rad`}
        />
      }
      caption="Play or scrub θ. The dashed ray is the direction at angle θ; the point sits on it at distance r = f(θ) from the origin, and the left plot traces the curve it draws. The right plot shows the same function as an ordinary graph of r against θ. Where r is negative (the rose, and the circle r = 2 cos θ past θ = π/2), the point lies on the opposite ray, behind the origin. The circle r = 2 cos θ is traced completely by θ ∈ [0, π]; the three-petal rose by θ ∈ [0, π] as well, and the second half of its sweep retraces it."
      readouts={
        <>
          <Readout label="θ" value={`${formatNumber(theta)} rad = ${formatNumber(toDegrees(theta))}°`} />
          <Readout label="r = f(θ)" value={formatNumber(r)} />
          <Readout label="x = r cos θ" value={formatNumber(p[0])} />
          <Readout label="y = r sin θ" value={formatNumber(p[1])} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div ref={cell}>
          <Plot x={px} y={py} fitHeight legend={false} title="the plane">
            {RINGS.map((c, i) => (
              <Curve key={i} name="polar grid" x={c.x} y={c.y} muted width={0.5} silent />
            ))}
            <Segments name="polar grid" segments={RAYS} muted width={0.5} />
            <Curve name="whole curve" x={full.x} y={full.y} muted dashed silent />
            <Segments name="ray at θ" segments={ray} slot={1} dashed />
            <Curve name="traced" x={traced.x} y={traced.y} slot={0} width={3} silent />
            <Vectors vectors={arrow} />
          </Plot>
        </div>
        <FixedHeight height={height}>
          <Plot x={tx} y={ry} legend={false} title="r against θ">
            <Curve name="r = f(θ)" x={full.t} y={full.r} muted />
            <Curve name="traced" x={traced.t} y={traced.r} slot={0} width={3} />
            <Points name="now" x={[theta]} y={[r]} emphasis size={8} live />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}

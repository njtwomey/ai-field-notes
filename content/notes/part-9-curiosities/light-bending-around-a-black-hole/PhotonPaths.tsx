import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, Shapes, useAxis, useFigureState } from 'aifn-render'
import {
  B_CRIT,
  deflection,
  HORIZON,
  ISCO,
  PHOTON_SPHERE,
  rayFromInfinity,
  rayPoints,
  strongDeflection,
  weakDeflection,
} from './blackHole'

const VIEW_X = 14
const VIEW_Y = 10
/** Ray points farther than this from the hole are not drawn. */
const R_DRAW = 40
/** The handle sits on the incoming asymptote at this x. */
const HANDLE_X = -12
const B_MIN = 0.1
const B_MAX = 9.5
/** The fan of background rays, drawn above and mirrored below the axis. */
const FAN = [1, 2, 3, 4, 5, 6, 7, 8, 9]
/** RK4 step in φ and the largest azimuth followed: three turns. */
const STEP = 0.01
const PHI_MAX = 6 * Math.PI

const circle = (r: number) => {
  const a = Array.from({ length: 181 }, (_, i) => (2 * Math.PI * i) / 180)
  return { x: a.map((v) => r * Math.cos(v)), y: a.map((v) => r * Math.sin(v)) }
}
const HORIZON_C = circle(HORIZON)
const PHOTON_C = circle(PHOTON_SPHERE)
const ISCO_C = circle(ISCO)
const HORIZON_SHAPE = [{ contours: [HORIZON_C.x.map((x, i) => [x, HORIZON_C.y[i]] as const)], tone: 'ink' as const }]

const deg = (a: number) => (Number.isFinite(a) ? `${((a * 180) / Math.PI).toFixed(1)}°` : '–')
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '–')

/**
 * Rays from the left past a Schwarzschild black hole, each integrated from the orbit equation u'' = 3u² − u with RK4
 * in φ: the dragged ray in colour over a fixed fan.
 */
export function PhotonPaths() {
  const state = useFigureState({
    b: float(5.3, {
      min: B_MIN,
      max: B_MAX,
      label: 'impact parameter b (units of M)',
      suggestions: [3, 5.2, 5.21, 5.3, 6, 9],
    }),
  })
  const b = state.b

  const fan = useMemo(
    () =>
      FAN.map((fb) => {
        const ray = rayFromInfinity(fb, STEP, PHI_MAX)
        return { b: fb, up: rayPoints(ray, R_DRAW, 1), down: rayPoints(ray, R_DRAW, -1) }
      }),
    [],
  )
  const ray = useMemo(() => rayFromInfinity(b, STEP, PHI_MAX), [b])
  const path = useMemo(() => rayPoints(ray, R_DRAW, 1), [ray])
  const exact = useMemo(() => deflection(b), [b])

  const xa = useAxis({ label: 'x (units of M)', range: [-VIEW_X, VIEW_X] })
  const ya = useAxis({ label: 'y', range: [-VIEW_Y, VIEW_Y], equal: xa })

  return (
    <Figure
      title="Photon paths past a black hole"
      state={state}
      defaultSize="L"
      aspect={1.4}
      readouts={{
        'the ray': (
          <>
            <Readout label="b − b_c (b_c = 3√3 ≈ 5.196)" value={fmt(b - B_CRIT)} />
            <Readout label="fate" value={exact.fate} />
            <Readout label="closest approach r₀" value={fmt(exact.closest)} />
            <Readout label="azimuth swept, in turns" value={fmt(exact.turns)} />
            <Readout label="first-integral error along the RK4 ray" value={fmt(ray.residual)} />
          </>
        ),
        deflection: (
          <>
            <Readout label="deflection α" value={`${deg(exact.alpha)} (${fmt(exact.alpha)} rad)`} />
            <Readout label="weak field 4/b" value={`${deg(weakDeflection(b))} (${fmt(weakDeflection(b))} rad)`} />
            <Readout label="strong field ln(3.4823 / (b − b_c))" value={fmt(strongDeflection(b))} />
          </>
        ),
      }}
      caption={
        <>
          Units G = c = M = 1, so the horizon (black disc) is at r = 2, the photon sphere (dashed) at r = 3 and the
          innermost stable circular orbit (grey) at r = 6. Rays arrive from the left, parallel to the x-axis, at height
          b, the impact parameter; each is the orbit equation u&apos;&apos; = 3u² − u integrated with RK4 in the azimuth
          φ. Drag the handle on the incoming ray, or type b, to move the coloured ray. Rays with b below b_c ≈ 5.196
          cross the horizon. Rays just above b_c circle the photon sphere before leaving: try b = 5.2 (one and a half
          turns) and b = 5.21. The readouts set the exact deflection against the weak-field value 4/b, which is good for
          large b, and the strong-field logarithm, which is good near b_c.
        </>
      }
    >
      <Plot x={xa} y={ya}>
        {fan.map((f) => (
          <Curve key={`u${f.b}`} name="fan of rays" x={f.up.x} y={f.up.y} muted thin silent />
        ))}
        {fan.map((f) => (
          <Curve key={`d${f.b}`} name="fan of rays" x={f.down.x} y={f.down.y} muted thin silent />
        ))}
        <Curve name="r = 6 (ISCO)" x={ISCO_C.x} y={ISCO_C.y} muted silent />
        <Curve name="photon sphere r = 3" x={PHOTON_C.x} y={PHOTON_C.y} emphasis dashed silent />
        <Shapes name="horizon r = 2" shapes={HORIZON_SHAPE} />
        <Curve name="incoming asymptote y = b" x={[-VIEW_X, 0]} y={[b, b]} muted dashed silent />
        <Curve name="the dragged ray" x={path.x} y={path.y} slot={0} silent />
        <Handle
          kind="point"
          at={[HANDLE_X, b]}
          label="b"
          slot={0}
          onDrag={([, y]) => state.set('b', Math.min(B_MAX, Math.max(B_MIN, Math.abs(y))))}
        />
      </Plot>
    </Figure>
  )
}

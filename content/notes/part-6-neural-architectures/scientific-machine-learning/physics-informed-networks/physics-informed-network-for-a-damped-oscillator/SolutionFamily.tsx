import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const OMEGA = 15
/** Where the velocity handle sits: a short step along the tangent at t = 0. */
const H = 0.06
const ts = Array.from({ length: 201 }, (_, i) => i / 200)

/**
 * The solution of x″ + 2ζω₀x′ + ω₀²x = 0 (underdamped) from x(0) = x₀ and x′(0) = v₀:
 * x(t) = e^{−ζω₀t}(x₀ cos ω_d t + ((v₀ + ζω₀x₀)/ω_d) sin ω_d t).
 */
function solution(zeta: number, x0: number, v0: number) {
  const wd = OMEGA * Math.sqrt(1 - zeta * zeta)
  const b = (v0 + zeta * OMEGA * x0) / wd
  return (t: number) => Math.exp(-zeta * OMEGA * t) * (x0 * Math.cos(wd * t) + b * Math.sin(wd * t))
}

/** The equation alone admits a two-parameter family of motions; the starting position and velocity pick one member. */
export function SolutionFamily() {
  const state = useFigureState({
    zeta: slider(0.02, 0.6, 0.2, { step: 0.02, label: 'damping ratio ζ' }),
    x0: slider(-1.2, 1.2, 1, { step: 0.01, label: 'x(0)' }),
    v0: slider(-20, 20, 0, { step: 0.1, label: "x'(0)" }),
  })
  const { zeta, x0, v0 } = state
  const view = useMemo(() => {
    // Other members of the family: the same equation from a fan of starting points and velocities.
    const family = [-1, -0.5, 0.5, 1].flatMap((a) => [-12, 0, 12].map((v) => ts.map(solution(zeta, a, v))))
    const chosen = ts.map(solution(zeta, x0, v0))
    const wd = OMEGA * Math.sqrt(1 - zeta * zeta)
    const b = (v0 + zeta * OMEGA * x0) / wd
    return { family, chosen, amplitude: Math.hypot(x0, b), phase: Math.atan2(b, x0), wd }
  }, [zeta, x0, v0])
  const t = useAxis({ label: 't', range: [0, 1] })
  const x = useAxis({ label: 'x', range: [-1.6, 1.6] })
  return (
    <Figure
      title="The equation fixes the shape, the conditions fix the member"
      state={state}
      hoverReadout={false}
      caption={`Grey: solutions of x″ + 2ζω₀x′ + ω₀²x = 0 with ω₀ = ${OMEGA}, from twelve different starting positions and velocities. All of them have zero residual. Coloured: the member through the chosen starting position and velocity. Drag the dot at t = 0 to set x(0) and the second dot to set the slope x′(0). The damping ratio sets how fast every member decays and how often it swings, the same for all of them; the two conditions set the amplitude and phase. The zero function is a member too, which is why a network trained on the residual alone collapses to x = 0.`}
      readouts={
        <>
          <Readout label="x(0), x′(0)" value={`${formatNumber(x0)}, ${formatNumber(v0)}`} />
          <Readout label="amplitude A" value={formatNumber(view.amplitude)} />
          <Readout label="phase φ" value={`${formatNumber(view.phase)} rad`} />
          <Readout label="damped frequency ω_d" value={formatNumber(view.wd)} />
        </>
      }
    >
      <Plot x={t} y={x}>
        {view.family.map((y, i) => (
          <Curve key={i} name={i === 0 ? 'other members' : `member ${i}`} x={ts} y={y} muted thin silent />
        ))}
        <Curve name="x = 0" x={[0, 1]} y={[0, 0]} muted dashed silent />
        <Curve name="chosen member" x={ts} y={view.chosen} slot={0} emphasis />
        <Curve name="initial slope" x={[0, H]} y={[x0, x0 + H * v0]} slot={2} thin silent />
        <Handle
          kind="point"
          at={[0, x0]}
          label="x(0)"
          onDrag={([, y]) => state.set('x0', Math.max(-1.2, Math.min(1.2, Math.round(y * 100) / 100)))}
        />
        <Handle
          kind="point"
          at={[H, x0 + H * v0]}
          label="x′(0)"
          onDrag={([, y]) => state.set('v0', Math.max(-20, Math.min(20, Math.round(((y - x0) / H) * 10) / 10)))}
        />
      </Plot>
    </Figure>
  )
}

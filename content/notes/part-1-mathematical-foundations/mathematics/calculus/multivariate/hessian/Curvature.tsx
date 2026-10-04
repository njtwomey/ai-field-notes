import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Vec = [number, number]

const AXIS = toFlat(linspace(-2.5, 2.5, 61))
const ARROW = 1.6
/** Width of the plateau: f(h) = tanh(q(h) / SCALE), q = ½ hᵀHh. */
const SCALE = 1.5

/**
 * A surface whose Hessian at the origin is H = Q diag(λ₁, λ₂) Qᵀ. The surface is f(h) = S·tanh(½hᵀHh / S): near the
 * origin it equals the quadratic form, and far away it levels off, so the colour shows the shape instead of washing out.
 */
export function Curvature() {
  const state = useFigureState({
    l1: float(1.5, { min: -2, max: 2, step: 0.05, label: 'eigenvalue λ₁' }),
    l2: float(0.5, { min: -2, max: 2, step: 0.05, label: 'eigenvalue λ₂' }),
    angle: slider(0, 180, 30, { step: 1, label: 'angle of q₁ (degrees)' }),
  })

  const r = useMemo(() => {
    const t = (state.angle * Math.PI) / 180
    const q1: Vec = [Math.cos(t), Math.sin(t)]
    const q2: Vec = [-Math.sin(t), Math.cos(t)]
    const [a, b] = [state.l1, state.l2]
    // H = a q1 q1ᵀ + b q2 q2ᵀ.
    const H = [
      [a * q1[0] * q1[0] + b * q2[0] * q2[0], a * q1[0] * q1[1] + b * q2[0] * q2[1]],
      [a * q1[1] * q1[0] + b * q2[1] * q2[0], a * q1[1] * q1[1] + b * q2[1] * q2[1]],
    ]
    const quad = (x: number, y: number) => 0.5 * (H[0][0] * x * x + 2 * H[0][1] * x * y + H[1][1] * y * y)
    const z = AXIS.map((y) => AXIS.map((x) => SCALE * Math.tanh(quad(x, y) / SCALE)))
    const eps = 1e-9
    const kind =
      a > eps && b > eps
        ? 'minimum'
        : a < -eps && b < -eps
          ? 'maximum'
          : (a > eps && b < -eps) || (a < -eps && b > eps)
            ? 'saddle'
            : 'degenerate: test inconclusive'
    return { q1, q2, H, z, kind }
  }, [state.l1, state.l2, state.angle])

  const vectors = [
    { from: [0, 0] as Vec, to: [ARROW * r.q1[0], ARROW * r.q1[1]] as Vec },
    { from: [0, 0] as Vec, to: [ARROW * r.q2[0], ARROW * r.q2[1]] as Vec },
  ]
  // The tip of the first eigenvector turns the eigenbasis; its angle is taken modulo 180°, like the slider.

  const xAxis = useAxis({ label: 'h₁' })
  const yAxis = useAxis({ label: 'h₂' })
  return (
    <Figure
      title="Curvature from the Hessian"
      state={state}
      caption="The surface has Hessian H at the origin, built from two eigenvalues λ₁, λ₂ and an eigenvector q₁ (drag its tip, or use the angle slider); q₂ is perpendicular. Red is above the value at the origin and blue below. Both eigenvalues positive: a minimum, red all round. Both negative: a maximum. Opposite signs: a saddle, rising along one eigenvector and falling along the other. A zero eigenvalue leaves a flat direction, where second derivatives cannot decide."

      readouts={
        <>
          <Readout
            label="H"
            value={`[[${formatNumber(r.H[0][0])}, ${formatNumber(r.H[0][1])}], [${formatNumber(r.H[1][0])}, ${formatNumber(r.H[1][1])}]]`}
          />
          <Readout label="det H" value={formatNumber(state.l1 * state.l2)} />
          <Readout label="trace H" value={formatNumber(state.l1 + state.l2)} />
          <Readout label="stationary point" value={r.kind} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-2xl">
        <Plot x={xAxis} y={yAxis} height={420}>
          <Raster x={AXIS} y={AXIS} z={r.z} scale={'diverging'} range={[-SCALE, SCALE]} valueLabel={'f − f(0)'} />
          <Vectors vectors={vectors} />
          <Handle
            kind="point"
            at={vectors[0].to}
            label="q₁"
            onDrag={([x, y]) => state.set('angle', ((((Math.atan2(y, x) * 180) / Math.PI) % 180) + 180) % 180)}
          />
        </Plot>
      </div>
    </Figure>
  )
}

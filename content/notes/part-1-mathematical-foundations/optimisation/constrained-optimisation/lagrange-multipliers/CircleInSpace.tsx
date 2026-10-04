import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, tField, type ConstrainedProblem } from './ConstrainedExplorer'

type Vec3 = [number, number, number]

const DEG = Math.PI / 180
/** The plane x₁ + x₂ + x₃ = 1 meets the unit sphere in a circle with this centre and radius. */
const CENTRE: Vec3 = [1 / 3, 1 / 3, 1 / 3]
const RADIUS = Math.sqrt(2 / 3)
/** An orthonormal basis of the plane: points are CENTRE + s₁U + s₂V. */
const U: Vec3 = [1 / Math.SQRT2, -1 / Math.SQRT2, 0]
const V: Vec3 = [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)]

const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const lift = (s1: number, s2: number): Vec3 => [0, 1, 2].map((i) => CENTRE[i] + s1 * U[i] + s2 * V[i]) as Vec3

/**
 * f(x) = ‖x − a‖² restricted to the plane, in the plane's coordinates s. With α the projection of a − CENTRE onto the
 * plane, f = ‖s − α‖² + (distance of a from the plane)², and the sphere is ‖s‖² = 2/3 within the plane.
 */
function problem(alpha: [number, number], offset: number): ConstrainedProblem {
  return {
    f: (s1, s2) => (s1 - alpha[0]) ** 2 + (s2 - alpha[1]) ** 2 + offset ** 2,
    gradF: (s1, s2) => [2 * (s1 - alpha[0]), 2 * (s2 - alpha[1])],
    gradG: (s1, s2) => [2 * s1, 2 * s2],
    curve: (deg) => [RADIUS * Math.cos(deg * DEG), RADIUS * Math.sin(deg * DEG)],
    tRange: [0, 360],
    closed: true,
    xRange: [-2, 2],
    yRange: [-2, 2],
    goal: 'min',
  }
}

/**
 * Least-squares multipliers for ∇f ≈ λ₁∇g₁ + λ₂∇g₂ in three dimensions, with ∇g₁ = 1 (the plane) and ∇g₂ = 2x (the
 * sphere), and the residual. The residual is zero exactly at a stationary point.
 */
function decompose(x: Vec3, a: Vec3) {
  const gf: Vec3 = [0, 1, 2].map((i) => 2 * (x[i] - a[i])) as Vec3
  const g1: Vec3 = [1, 1, 1]
  const g2: Vec3 = [2 * x[0], 2 * x[1], 2 * x[2]]
  const [p, q, r] = [dot3(g1, g1), dot3(g1, g2), dot3(g2, g2)]
  const [b1, b2] = [dot3(g1, gf), dot3(g2, gf)]
  const det = p * r - q * q
  const l1 = (r * b1 - q * b2) / det
  const l2 = (p * b2 - q * b1) / det
  const res = [0, 1, 2].map((i) => gf[i] - l1 * g1[i] - l2 * g2[i])
  return { gf, l1, l2, residual: Math.hypot(...res) }
}

/** Rounds float noise such as 1e-17 to 0. */
const fmt3 = (v: Vec3) => `(${v.map((x) => formatNumber(Math.abs(x) < 1e-9 ? 0 : x)).join(', ')})`

export function CircleInSpace() {
  const state = useFigureState({
    t: tField([0, 360], 140, 0.5, 'angle θ (degrees)', (v) => `${formatNumber(v)}°`),
    a1: slider(-2, 2, 2, { step: 0.05, label: 'a₁' }),
    a2: slider(-2, 2, 1, { step: 0.05, label: 'a₂' }),
    a3: slider(-2, 2, 0, { step: 0.05, label: 'a₃' }),
  })
  const a: Vec3 = [state.a1, state.a2, state.a3]
  const rel: Vec3 = [a[0] - CENTRE[0], a[1] - CENTRE[1], a[2] - CENTRE[2]]
  const alpha0 = dot3(rel, U)
  const alpha1 = dot3(rel, V)
  const offset = (rel[0] + rel[1] + rel[2]) / Math.sqrt(3)
  const p = useMemo(() => problem([alpha0, alpha1], offset), [alpha0, alpha1, offset])
  const markers = useMemo(
    () => ({ name: 'a projected onto the plane', points: [[alpha0, alpha1]] as [number, number][] }),
    [alpha0, alpha1],
  )
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="Two constraints: the closest point to a on a circle in space"
      caption="The plane x₁ + x₂ + x₃ = 1 meets the unit sphere in a circle. Left: the plane seen face on, with contours of f(x) = ‖x − a‖² (grey), the circle, and the in-plane parts of ∇f (arrow) and of the sphere's normal (thin line); the plane's own normal points out of the page. Drag the point around the circle, and move a with the sliders. Right: f along the circle. The readouts fit ∇f by λ₁∇g₁ + λ₂∇g₂ in three dimensions; the residual vanishes only at the closest and farthest points."
      xLabel="s₁ along (1, −1, 0)/√2"
      yLabel="s₂ along (1, 1, −2)/√6"
      tLabel="angle θ (degrees)"
      tSymbol="θ"
      fLabel="f"
      formatT={(v) => `${formatNumber(v)}°`}
      markers={markers}
      readout={({ point }) => {
        const x = lift(point[0], point[1])
        const d = decompose(x, a)
        return (
          <>
            <Readout label="x" value={fmt3(x)} />
            <Readout label="∇f = 2(x − a)" value={fmt3(d.gf)} />
            <Readout label="λ₁ (plane)" value={formatNumber(d.l1)} />
            <Readout label="λ₂ (sphere)" value={formatNumber(d.l2)} />
            <Readout label="‖∇f − λ₁∇g₁ − λ₂∇g₂‖" value={formatNumber(d.residual)} />
          </>
        )
      }}
    />
  )
}

import { useMemo } from 'react'
import { basisPursuit } from 'aifn-compute/signal/sparse'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const R = 3
const THETA = toFlat(linspace(0, 2 * Math.PI, 241))
/** Entries this close to zero count as zero in the readouts. */
const ZERO = 1e-6

const coord = (initial: number, label: string) => slider(-2.5, 2.5, initial, { step: 0.05, label, onChart: true })
const nonZeros = (x: readonly number[]) => x.filter((v) => Math.abs(v) > ZERO).length
const show = (x: readonly number[]) => `(${x.map((v) => formatNumber(Math.abs(v) < ZERO ? 0 : v)).join(', ')})`

/**
 * The solutions of one equation a·x = b in two unknowns form a line. The smallest ℓ2 ball touches it at the foot of the
 * perpendicular from the origin, which is dense; the smallest ℓ1 ball, a diamond, touches it at a corner on an axis.
 */
export function NormBalls() {
  const state = useFigureState({
    px: coord(1.4, 'x₁ of the ℓ2 solution'),
    py: coord(0.6, 'x₂ of the ℓ2 solution'),
    half: setting(false, 'show the ℓ½ ball'),
  })
  const { px, py, half } = state

  const g = useMemo(() => {
    // The line is {x : p·x = |p|²}, perpendicular to p through p, so p is its point of least ℓ2 norm.
    // At the origin the line is undefined, so a handle dragged there is held a little to its right.
    const p = Math.hypot(px, py) < 0.05 ? [0.05, 0] : [px, py]
    const norm = Math.hypot(p[0], p[1])
    const [ux, uy] = [p[0] / norm, p[1] / norm]
    const b = norm * norm
    const t = [-4 * R, 4 * R]
    const line = { x: t.map((s) => p[0] - s * uy), y: t.map((s) => p[1] + s * ux) }
    const l1 = toFlat(basisPursuit([[p[0], p[1]]], [b], { method: 'interior-point' }).x)
    // Every ball is drawn at the size at which it first meets the line.
    const r1 = b / Math.max(Math.abs(p[0]), Math.abs(p[1]))
    const circle = { x: THETA.map((s) => norm * Math.cos(s)), y: THETA.map((s) => norm * Math.sin(s)) }
    const diamond = { x: [r1, 0, -r1, 0, r1], y: [0, r1, 0, -r1, 0] }
    // |x₁|^½ + |x₂|^½ = r^½ is traced by |x₁| = r cos⁴θ, |x₂| = r sin⁴θ; it meets the line at the same corner.
    const quarter = {
      x: THETA.map((s) => r1 * Math.sign(Math.cos(s)) * Math.cos(s) ** 4),
      y: THETA.map((s) => r1 * Math.sign(Math.sin(s)) * Math.sin(s) ** 4),
    }
    const tie = Math.abs(Math.abs(p[0]) - Math.abs(p[1])) < 0.02 * norm
    return { p, line, l1: [l1[0], l1[1]], circle, diamond, quarter, tie }
  }, [px, py])

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Why the ℓ1 ball finds a sparse solution"
      state={state}
      caption="One equation in two unknowns: every point of the line solves it. The ℓ2 solution (the handle; drag it) is the foot of the perpendicular from the origin, where the smallest circle meets the line; both of its coordinates are non-zero. The smallest ℓ1 ball, a diamond, first meets the line at one of its corners, which lie on the axes, so the ℓ1 solution has one non-zero coordinate. The corner it meets is on the axis of the larger coordinate of the ℓ2 solution. When the two coordinates have equal size the line is parallel to an edge of the diamond and every point of that edge is an ℓ1 solution. The ℓ½ ball is pointier still and meets the line at the same corner."
      readouts={
        <>
          <Readout label="ℓ2 solution" value={show(g.p)} />
          <Readout label="its non-zeros" value={String(nonZeros(g.p))} />
          <Readout label="ℓ1 solution" value={g.tie ? 'an edge (not unique)' : show(g.l1)} />
          <Readout label="its non-zeros" value={g.tie ? '1 or 2' : String(nonZeros(g.l1))} />
          <Readout
            label="‖x‖₁ of each"
            value={`${formatNumber(Math.abs(g.p[0]) + Math.abs(g.p[1]))} vs ${formatNumber(Math.abs(g.l1[0]) + Math.abs(g.l1[1]))}`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Curve name="solutions of a·x = b" x={g.line.x} y={g.line.y} emphasis />
          <Curve name="smallest ℓ2 ball" x={g.circle.x} y={g.circle.y} slot={0} dashed />
          <Curve name="smallest ℓ1 ball" x={g.diamond.x} y={g.diamond.y} slot={1} />
          {half && <Curve name="smallest ℓ½ ball" x={g.quarter.x} y={g.quarter.y} slot={2} dashed />}
          <Points name="ℓ1 solution" x={[g.l1[0]]} y={[g.l1[1]]} slot={1} size={10} />
          <Handle {...state.handle(['px', 'py'], { label: 'ℓ2 solution' })} />
        </Plot>
      </div>
    </Figure>
  )
}

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
  useFigureState,
} from 'aifn-render'
import { chebyshevScalarisation, simplexLattice } from 'aifn-methods/optimisation/multi-objective'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'

/** A two-field handle is a point handle, which takes a marker symbol. */
const asPoint = <H extends { kind: string }>(h: H) => h as Extract<H, { kind: 'point' }>

/** The two outer points of the worked example's front: (validation error, latency in ms). */
const P1 = [0.1, 80] as const
const P3 = [0.25, 20] as const
const NAMES = ['p₁', 'p₂′', 'p₃']
const X_MAX = 0.35
const Y_MAX = 120

const argmin = (v: readonly number[]) => v.reduce((best, x, i) => (x < v[best] ? i : best), 0)

/**
 * Three mutually non-dominated outcomes, the middle one movable. Weights λ ∝ (1/a₁, 1/a₂) aim along the ray from the
 * origin through a: the augmented Chebyshev scalarisation picks the point its L-shaped level set meets first, the
 * weighted sum the point its straight level set meets first. A middle point above the chord p₁p₃ is reachable only
 * by the first.
 */
export function ChebyshevReach() {
  const state = useFigureState({
    m1: slider(0.11, 0.24, 0.2, { step: 0.005, onChart: true, label: 'p₂′ error' }),
    m2: slider(21, 79, 55, { step: 1, onChart: true, label: 'p₂′ latency' }),
    a1: slider(0.01, X_MAX, 0.24, { step: 0.005, onChart: true, label: 'aim error' }),
    a2: slider(1, Y_MAX, 66, { step: 1, onChart: true, label: 'aim latency' }),
  })
  const { m1, m2, a1, a2 } = state
  const points = [P1, [m1, m2] as const, P3]

  // Weights proportional to 1/a, normalised to sum to 1.
  const raw = [1 / a1, 1 / a2]
  const weights = raw.map((w) => w / (raw[0] + raw[1]))
  const cheb = Array.from(toFlat(chebyshevScalarisation(points, weights, { rho: 0.05 })))
  const maxTerm = Array.from(toFlat(chebyshevScalarisation(points, weights, { rho: 0 })))
  const sums = points.map(([y1, y2]) => weights[0] * y1 + weights[1] * y2)
  const chebWinner = argmin(cheb)
  const sumWinner = argmin(sums)

  // ParEGO's 11 weight vectors on the objectives normalised by the ideal and nadir points of the three outcomes.
  const lattice = (() => {
    const ideal = [Math.min(...points.map((p) => p[0])), Math.min(...points.map((p) => p[1]))]
    const nadir = [Math.max(...points.map((p) => p[0])), Math.max(...points.map((p) => p[1]))]
    const range = [nadir[0] - ideal[0], nadir[1] - ideal[1]]
    const wins = { cheb: [0, 0, 0], sum: [0, 0, 0] }
    for (const w of toArray(simplexLattice(2, 10)) as number[][]) {
      wins.cheb[argmin(Array.from(toFlat(chebyshevScalarisation(points, w, { ideal, nadir }))))]++
      wins.sum[
        argmin(points.map((p) => w[0] * ((p[0] - ideal[0]) / range[0]) + w[1] * ((p[1] - ideal[1]) / range[1])))
      ]++
    }
    return wins
  })()
  const tally = (counts: number[]) => counts.map((c, i) => `${NAMES[i]} ${c}`).join(', ')

  // Level sets through each winner: the corner of max(λ₁y₁, λ₂y₂) = s sits on the aim ray at (s/λ₁, s/λ₂).
  const s = maxTerm[chebWinner]
  const corner = [s / weights[0], s / weights[1]] as const
  const t = sums[sumWinner]
  const rayEnd: [number, number] = (X_MAX * a2) / a1 <= Y_MAX ? [X_MAX, (X_MAX * a2) / a1] : [(Y_MAX * a1) / a2, Y_MAX]

  const x = useAxis({ label: 'validation error f₁', range: [0, X_MAX] })
  const y = useAxis({ label: 'latency f₂ (ms)', range: [0, Y_MAX] })
  return (
    <Figure
      title="A Chebyshev scalarisation reaches a point no weighted sum can"
      state={state}
      caption="Three non-dominated outcomes, error against latency. Drag the aim point a (ink diamond): the weights are λ ∝ (1/a₁, 1/a₂), so the aim sets a direction from the origin (dashed ray). The augmented Chebyshev scalarisation max(λ₁y₁, λ₂y₂) + 0.05(λ₁y₁ + λ₂y₂) has L-shaped level sets with their corner on the ray (blue), and it picks the first point the L meets. The weighted sum λ₁y₁ + λ₂y₂ has straight level sets (orange) and picks the first point a straight line meets. With p₂′ = (0.20, 55) above the chord from p₁ to p₃ (grey), aiming at p₂′ makes the Chebyshev function pick it, and no aim makes the weighted sum pick it. Drag p₂′ below the chord and the weighted sum can reach it too. The last two readouts run ParEGO's 11 evenly spaced weight vectors on the objectives rescaled to [0, 1]: with p₂′ at its starting place, the Chebyshev function picks it for 2 of them and the weighted sum for none."
      readouts={
        <>
          <Readout label="weights λ" value={`(${formatNumber(weights[0])}, ${formatNumber(weights[1])})`} />
          <Readout label="Chebyshev picks" value={NAMES[chebWinner]} />
          <Readout label="weighted sum picks" value={NAMES[sumWinner]} />
          <Readout label="Chebyshev scores of p₁, p₂′, p₃" value={cheb.map((v) => formatNumber(v)).join(', ')} />
          <Readout label="ParEGO's 11 weights, Chebyshev picks" value={tally(lattice.cheb)} />
          <Readout label="ParEGO's 11 weights, weighted sum picks" value={tally(lattice.sum)} />
        </>
      }
    >
      <Plot x={x} y={y} height={420}>
        <Curve name="chord p₁p₃" x={[P1[0], P3[0]]} y={[P1[1], P3[1]]} muted dashed />
        <Segments name="aim ray" segments={[{ from: [0, 0], to: rayEnd }]} emphasis dashed width={1} />
        <Curve
          name="Chebyshev level set"
          x={[corner[0], corner[0], X_MAX]}
          y={[Y_MAX, corner[1], corner[1]]}
          slot={0}
        />
        <Curve name="weighted-sum level set" x={[t / weights[0], 0]} y={[0, t / weights[1]]} slot={1} />
        <Points
          name="outcomes"
          x={points.map((p) => p[0])}
          y={points.map((p) => p[1])}
          labels={NAMES}
          slot={2}
          size={10}
        />
        <Handle {...state.handle(['m1', 'm2'], { label: 'p₂′' })} />
        <Handle {...asPoint(state.handle(['a1', 'a2'], { label: 'a' }))} symbol="diamond" />
      </Plot>
    </Figure>
  )
}

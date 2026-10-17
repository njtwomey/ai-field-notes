import { useMemo } from 'react'
import {
  Contours,
  Figure,
  formatNumber,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { lagrangePoints, linspace, potentialGrid, ROUTH_MU } from './restricted'

const XS = linspace(-1.6, 1.6, 241)
const YS = linspace(-1.3, 1.3, 197)

export function EffectivePotential() {
  const state = useFigureState({
    mu: slider(0.001, 0.5, 0.01, { step: 0.001, label: 'mass ratio μ = m₂ / (m₁ + m₂)' }),
  })
  const { mu } = state

  const points = useMemo(() => lagrangePoints(mu), [mu])
  const [l1, l2, l3, l4] = points
  // Colour 2Ω from its lowest value (at L4 and L5) to a little above its value at L1; higher values are capped.
  const cap = l1.C + 0.4
  const grid = useMemo(() => potentialGrid(mu, XS, YS, cap), [mu, cap])
  const levels = [l1.C, l2.C, l3.C].sort((a, b) => a - b)

  const xa = useAxis({ label: 'x (rotating frame)', range: [-1.6, 1.6], nice: false })
  const ya = useAxis({ label: 'y', range: [-1.3, 1.3], nice: false, equal: xa })

  const stable = 27 * mu * (1 - mu) < 1

  return (
    <Figure
      title="Effective potential and the five Lagrange points"
      state={state}
      defaultSize="L"
      caption="Colour: 2Ω(x, y) in the frame rotating with the two masses (m₁ at (−μ, 0), m₂ at (1 − μ, 0), separation 1, G = 1, total mass 1), capped above its value at L1. Contours: the levels of 2Ω through L1, L2 and L3, which are the zero-velocity curves of a particle whose Jacobi constant equals the value at that point. L1, L2 and L3 sit on the axis where ∂Ω/∂x = 0, found by Brent's method; L4 and L5 sit at the apexes of the two equilateral triangles on the line between the masses. Slide the mass ratio: L1 and L2 close in on m₂ as μ falls, at a distance near the Hill radius (μ/3)^(1/3)."
      readouts={
        <>
          <Readout label="L1, L2, L3 at x" value={[l1.x, l2.x, l3.x].map((v) => v.toFixed(4)).join(', ')} />
          <Readout label="2Ω at L1, L2, L3, L4" value={[l1.C, l2.C, l3.C, l4.C].map((v) => v.toFixed(4)).join(', ')} />
          <Readout label="Hill radius (μ/3)^(1/3)" value={formatNumber(Math.cbrt(mu / 3))} />
          <Readout
            label="27μ(1 − μ)"
            value={`${formatNumber(27 * mu * (1 - mu))}: L4, L5 ${stable ? 'stable' : 'unstable'} (μ ${stable ? '<' : '>'} ${ROUTH_MU.toFixed(4)})`}
          />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={XS}
          y={YS}
          z={grid}
          scale="sequential"
          range={[l4.C - 0.05, cap]}
          valueLabel="2Ω"
          fillOpacity={0.75}
        />
        <Contours x={XS} y={YS} z={grid} levels={levels} labels={false} />
        <Points name="masses" x={[-mu, 1 - mu]} y={[0, 0]} labels={['m₁', 'm₂']} size={12} muted />
        <Points
          name="Lagrange points"
          x={points.map((p) => p.x)}
          y={points.map((p) => p.y)}
          labels={points.map((p) => p.name)}
          emphasis
          size={8}
        />
      </Plot>
    </Figure>
  )
}

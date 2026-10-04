import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalPdf } from 'aifn/numerics/special'

const MU_MAX = 5
const MU_GRID = toFlat(linspace(0, MU_MAX, 101))
const X_GRID = toFlat(linspace(-6, MU_MAX + 6, 601))

/** Jensen–Shannon divergence in nats between N(0, 1) and N(μ, 1), by the trapezoidal rule on a fine grid. */
function jensenShannon(mu: number): number {
  const dx = X_GRID[1] - X_GRID[0]
  let total = 0
  for (const x of X_GRID) {
    const p = normalPdf(x)
    const q = normalPdf(x - mu)
    const m = 0.5 * (p + q)
    if (p > 0) total += 0.5 * p * Math.log(p / m)
    if (q > 0) total += 0.5 * q * Math.log(q / m)
  }
  return total * dx
}

// Closed forms for two unit-variance Gaussians a distance μ apart.
const kl = (mu: number) => (mu * mu) / 2
const totalVariation = (mu: number) => 2 * normalCdf(mu / 2) - 1
const hellingerSq = (mu: number) => 1 - Math.exp((-mu * mu) / 8)
const JS_CURVE = MU_GRID.map(jensenShannon)

/**
 * Divergences between N(0, 1) and N(μ, 1) as the second distribution moves away. KL grows without bound; Jensen–Shannon
 * saturates at log 2, and total variation and squared Hellinger at 1, once the distributions stop overlapping.
 */
export function DivergenceGrowth() {
  const state = useFigureState({
    mu: float(1.5, { min: 0, max: MU_MAX, step: 0.05, label: 'separation μ' }),
  })

  const densities = useMemo(() => {
    const xs = X_GRID.filter((_, i) => i % 3 === 0)
    return [
      { name: 'p = N(0, 1)', x: xs, y: xs.map((v: number) => normalPdf(v)), slot: 0 },
      { name: 'q = N(μ, 1)', x: xs, y: xs.map((x) => normalPdf(x - state.mu)), slot: 1 },
      {
        name: 'mixture m = (p + q)/2',
        x: xs,
        y: xs.map((x) => 0.5 * (normalPdf(x) + normalPdf(x - state.mu))),
        dashed: true,
        muted: true,
      },
    ] as const
  }, [state.mu])

  const curves = [
    { name: 'KL(p ‖ q)', x: MU_GRID, y: MU_GRID.map(kl), slot: 0 },
    { name: 'Jensen–Shannon', x: MU_GRID, y: JS_CURVE, slot: 1 },
    { name: 'total variation', x: MU_GRID, y: MU_GRID.map(totalVariation), slot: 2 },
    { name: 'squared Hellinger', x: MU_GRID, y: MU_GRID.map(hellingerSq), slot: 3 },
    { name: 'log 2', x: [0, MU_MAX], y: [Math.LN2, Math.LN2], dashed: true, muted: true },
  ] as const
  // μ is a location on both charts' horizontal axes, so dragging it moves q.
  const onDrag = (x: number) => state.set('mu', x)
  const js = jensenShannon(state.mu)

  const xAxis = useAxis({ label: 'x', range: [-4, MU_MAX + 4] })
  const yAxis = useAxis({ label: 'density', range: [0, 0.45] })
  const xAxis2 = useAxis({ label: 'separation μ', range: [0, MU_MAX] })
  const yAxis2 = useAxis({ label: 'divergence', range: [0, 2.5] })
  return (
    <Figure
      title="Bounded and unbounded divergences"
      state={state}
      caption="p = N(0, 1) is fixed and q = N(μ, 1) moves away from it. Left: the two densities and their mixture m, which the Jensen–Shannon divergence compares both against. Right: four divergences against μ. KL grows as μ²/2 without limit. Jensen–Shannon levels off at log 2 ≈ 0.693 nats, and total variation and squared Hellinger at 1, once the densities no longer overlap. Drag μ on either chart."

      readouts={
        <>
          <Readout label="KL" value={formatNumber(kl(state.mu))} />
          <Readout label="Jensen–Shannon" value={formatNumber(js)} />
          <Readout label="total variation" value={formatNumber(totalVariation(state.mu))} />
          <Readout label="squared Hellinger" value={formatNumber(hellingerSq(state.mu))} />
          <Readout label="χ²" value={formatNumber(Math.expm1(state.mu * state.mu))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...densities[0]} />
          <Curve {...densities[1]} />
          <Curve {...densities[2]} />
          <Handle kind="x" at={state.mu} label="μ" onDrag={onDrag} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...curves[0]} />
          <Curve {...curves[1]} />
          <Curve {...curves[2]} />
          <Curve {...curves[3]} />
          <Curve {...curves[4]} />
          <Handle kind="x" at={state.mu} label="μ" onDrag={onDrag} />
        </Plot>
      </div>
    </Figure>
  )
}

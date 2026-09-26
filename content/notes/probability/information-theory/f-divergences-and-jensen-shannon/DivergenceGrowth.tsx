import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'

const MU_MAX = 5
const MU_GRID = linspace(0, MU_MAX, 101)
const X_GRID = linspace(-6, MU_MAX + 6, 601)

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
  const mu = useParam(1.5, { min: 0, max: MU_MAX, step: 0.05 })

  const densities = useMemo((): XYSeries[] => {
    const xs = X_GRID.filter((_, i) => i % 3 === 0)
    return [
      { name: 'p = N(0, 1)', type: 'line', x: xs, y: xs.map(normalPdf), slot: 0 },
      { name: 'q = N(μ, 1)', type: 'line', x: xs, y: xs.map((x) => normalPdf(x - mu.value)), slot: 1 },
      {
        name: 'mixture m = (p + q)/2',
        type: 'line',
        x: xs,
        y: xs.map((x) => 0.5 * (normalPdf(x) + normalPdf(x - mu.value))),
        dashed: true,
        muted: true,
      },
    ]
  }, [mu.value])

  const curves: XYSeries[] = [
    { name: 'KL(p ‖ q)', type: 'line', x: MU_GRID, y: MU_GRID.map(kl), slot: 0 },
    { name: 'Jensen–Shannon', type: 'line', x: MU_GRID, y: JS_CURVE, slot: 1 },
    { name: 'total variation', type: 'line', x: MU_GRID, y: MU_GRID.map(totalVariation), slot: 2 },
    { name: 'squared Hellinger', type: 'line', x: MU_GRID, y: MU_GRID.map(hellingerSq), slot: 3 },
    { name: 'log 2', type: 'line', x: [0, MU_MAX], y: [Math.LN2, Math.LN2], dashed: true, muted: true },
  ]
  // μ is a location on both charts' horizontal axes, so dragging it moves q.
  const onDrag = (x: number) => mu.set(x)
  const densityHandles: Handle[] = [{ kind: 'x', at: mu.value, label: 'μ', onDrag }]
  const curveHandles: Handle[] = [{ kind: 'x', at: mu.value, label: 'μ', onDrag }]
  const js = jensenShannon(mu.value)

  return (
    <Interactive
      title="Bounded and unbounded divergences"
      caption="p = N(0, 1) is fixed and q = N(μ, 1) moves away from it. Left: the two densities and their mixture m, which the Jensen–Shannon divergence compares both against. Right: four divergences against μ. KL grows as μ²/2 without limit. Jensen–Shannon levels off at log 2 ≈ 0.693 nats, and total variation and squared Hellinger at 1, once the densities no longer overlap. Drag μ on either chart."
      controls={<ParamSlider label="separation μ" param={mu} />}
      readout={
        <>
          <Readout label="KL" value={formatNumber(kl(mu.value))} />
          <Readout label="Jensen–Shannon" value={formatNumber(js)} />
          <Readout label="total variation" value={formatNumber(totalVariation(mu.value))} />
          <Readout label="squared Hellinger" value={formatNumber(hellingerSq(mu.value))} />
          <Readout label="χ²" value={formatNumber(Math.expm1(mu.value * mu.value))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={densities}
          xLabel="x"
          yLabel="density"
          xRange={[-4, MU_MAX + 4]}
          yRange={[0, 0.45]}
          handles={densityHandles}
          height={300}
        />
        <XYChart
          series={curves}
          xLabel="separation μ"
          yLabel="divergence"
          xRange={[0, MU_MAX]}
          yRange={[0, 2.5]}
          handles={curveHandles}
          height={300}
        />
      </div>
    </Interactive>
  )
}

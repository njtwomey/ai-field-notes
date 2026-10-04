import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalPdf, normalQuantile } from 'aifn/numerics/special'

/**
 * One observation X, H₀: X ~ N(0, 1) against H₁: X ~ N(0, σ₁²). The likelihood ratio depends on x only through x², so
 * the most powerful test rejects for large |x| when σ₁ > 1 and for small |x| when σ₁ < 1.
 */
export function LikelihoodRatioRegion() {
  const state = useFigureState({
    sigma: float(2, { min: 0.3, max: 3, step: 0.05, label: 'σ₁ under H₁' }),
    alpha: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'significance level α' }),
  })

  const result = useMemo(() => {
    const wider = state.sigma > 1
    // Critical value on |x|: P₀(|X| ≥ c) = α when σ₁ > 1, P₀(|X| ≤ c) = α when σ₁ < 1.
    const c = wider ? normalQuantile(1 - state.alpha / 2) : normalQuantile((1 + state.alpha) / 2)
    const h1 = (x: number) => normalPdf(x / state.sigma) / state.sigma
    const lim = Math.max(4, 4 * state.sigma)
    const xs = toFlat(linspace(-lim, lim, 600))
    const region = (keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map(h1) }
    }
    const series: SeriesSpec[] = [
      { name: 'H₀: N(0, 1)', type: 'line', x: xs, y: xs.map((v: number) => normalPdf(v)), slot: 0 },
      { name: 'H₁: N(0, σ₁²)', type: 'line', x: xs, y: xs.map(h1), slot: 1 },
      ...(wider ? [region((x) => x <= -c), region((x) => x >= c)] : [region((x) => Math.abs(x) <= c)]).map(
        (r): SeriesSpec => ({ name: 'power of the LR test', type: 'line', ...r, slot: 1, area: true }),
      ),
    ]
    const lrPower = wider ? 2 * (1 - normalCdf(c / state.sigma)) : 2 * normalCdf(c / state.sigma) - 1
    const upperPower = 1 - normalCdf(normalQuantile(1 - state.alpha) / state.sigma)
    return { series, lrPower, upperPower, c, wider }
  }, [state.sigma, state.alpha])

  const xAxis = useAxis({ label: 'observation x', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="The likelihood ratio chooses the critical region"
      state={state}
      caption="Under H₁ the observation has standard deviation σ₁ instead of 1. The shaded area is the most powerful critical region at level α, weighted by the H₁ density: its area is the power. When σ₁ > 1 the region is both tails; when σ₁ < 1 it is a narrow band around zero. An upper-tailed test at the same level is much weaker."

      readouts={
        <>
          <Readout
            label="LR critical region"
            value={result.wider ? `|x| ≥ ${formatNumber(result.c)}` : `|x| ≤ ${formatNumber(result.c)}`}
          />
          <Readout label="power, LR test" value={formatNumber(result.lrPower)} />
          <Readout label="power, upper-tailed test" value={formatNumber(result.upperPower)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(result.series)}
      </Plot>
    </Figure>
  )
}

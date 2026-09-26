import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf, normalQuantile } from '@/lib/math/special'

/**
 * One observation X, H₀: X ~ N(0, 1) against H₁: X ~ N(0, σ₁²). The likelihood ratio depends on x only through x², so
 * the most powerful test rejects for large |x| when σ₁ > 1 and for small |x| when σ₁ < 1.
 */
export function LikelihoodRatioRegion() {
  const [sigma, setSigma] = useState(2)
  const [alpha, setAlpha] = useState(0.05)

  const result = useMemo(() => {
    const wider = sigma > 1
    // Critical value on |x|: P₀(|X| ≥ c) = α when σ₁ > 1, P₀(|X| ≤ c) = α when σ₁ < 1.
    const c = wider ? normalQuantile(1 - alpha / 2) : normalQuantile((1 + alpha) / 2)
    const h1 = (x: number) => normalPdf(x / sigma) / sigma
    const lim = Math.max(4, 4 * sigma)
    const xs = linspace(-lim, lim, 600)
    const region = (keep: (x: number) => boolean) => {
      const inside = xs.filter(keep)
      return { x: inside, y: inside.map(h1) }
    }
    const series: XYSeries[] = [
      { name: 'H₀: N(0, 1)', type: 'line', x: xs, y: xs.map(normalPdf), slot: 0 },
      { name: 'H₁: N(0, σ₁²)', type: 'line', x: xs, y: xs.map(h1), slot: 1 },
      ...(wider ? [region((x) => x <= -c), region((x) => x >= c)] : [region((x) => Math.abs(x) <= c)]).map(
        (r): XYSeries => ({ name: 'power of the LR test', type: 'line', ...r, slot: 1, area: true }),
      ),
    ]
    const lrPower = wider ? 2 * (1 - normalCdf(c / sigma)) : 2 * normalCdf(c / sigma) - 1
    const upperPower = 1 - normalCdf(normalQuantile(1 - alpha) / sigma)
    return { series, lrPower, upperPower, c, wider }
  }, [sigma, alpha])

  return (
    <Interactive
      title="The likelihood ratio chooses the critical region"
      caption="Under H₁ the observation has standard deviation σ₁ instead of 1. The shaded area is the most powerful critical region at level α, weighted by the H₁ density: its area is the power. When σ₁ > 1 the region is both tails; when σ₁ < 1 it is a narrow band around zero. An upper-tailed test at the same level is much weaker."
      controls={
        <>
          <ParamSlider label="σ₁ under H₁" value={sigma} onChange={setSigma} min={0.3} max={3} step={0.05} />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.2}
            step={0.01}
          />
        </>
      }
      readout={
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
      <XYChart height={300} series={result.series} xLabel="observation x" yLabel="density" yRange={[0, undefined]} />
    </Interactive>
  )
}

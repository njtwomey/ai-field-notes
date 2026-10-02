import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { binomialPmf, binomialUpper } from '@/lib/math/tests'

const P0 = 0.01

/**
 * The quality-control test of H₀: p = 1% against H₁: p > 1%. S ~ Binom(n, p) counts defects; the critical value c is
 * the smallest count whose upper tail under H₀ is at most α.
 */
export function CriticalRegion() {
  const [n, setN] = useState(200)
  const [alpha, setAlpha] = useState(0.05)
  const [p1, setP1] = useState(0.02)

  const result = useMemo(() => {
    let c = 0
    while (binomialUpper(c, n, P0) > alpha) c++
    const top = Math.ceil(n * p1 + 4 * Math.sqrt(n * p1 * (1 - p1)) + 4)
    const ks = Array.from({ length: top + 1 }, (_, k) => k)
    const kept = ks.filter((k) => k < c)
    const rejected = ks.filter((k) => k >= c)
    const series: XYSeries[] = [
      { name: 'H₀ pmf, do not reject', type: 'bar', x: kept, y: kept.map((k) => binomialPmf(k, n, P0)), muted: true },
      { name: 'H₀ pmf, reject', type: 'bar', x: rejected, y: rejected.map((k) => binomialPmf(k, n, P0)), slot: 0 },
      { name: 'H₁ pmf', type: 'line', x: ks, y: ks.map((k) => binomialPmf(k, n, p1)), slot: 1, dashed: true },
    ]
    return { c, series, size: binomialUpper(c, n, P0), power: binomialUpper(c, n, p1) }
  }, [n, alpha, p1])

  return (
    <Interactive
      title="Critical region for a defect rate"
      caption="Grey and blue bars are the distribution of the defect count S when the defect rate is 1%. The blue bars, S ≥ c, form the critical region: its total probability is the size of the test, at most α. The dashed line is the distribution of S under an alternative defect rate. Its mass over the critical region is the power."
      controls={
        <>
          <ParamSlider label="items inspected n" value={n} onChange={setN} min={20} max={1500} step={10} />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.2}
            step={0.01}
          />
          <ParamSlider
            label="alternative defect rate"
            value={p1}
            onChange={setP1}
            min={0.011}
            max={0.06}
            step={0.001}
            format={(v) => `${(100 * v).toFixed(1)}%`}
          />
        </>
      }
      readout={
        <>
          <Readout label="critical value c" value={result.c} />
          <Readout label="size P₀(S ≥ c)" value={formatNumber(result.size)} />
          <Readout label="power P₁(S ≥ c)" value={formatNumber(result.power)} />
        </>
      }
    >
      <XYChart height={300} series={result.series} xLabel="defects S" yLabel="probability" yRange={[0, undefined]} />
    </Interactive>
  )
}

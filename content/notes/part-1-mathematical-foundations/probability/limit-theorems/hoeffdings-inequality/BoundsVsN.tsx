import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { logChoose } from '@/lib/math/special'

const NS = Array.from({ length: 60 }, (_, i) => 10 * (i + 1))
const FLOOR = 1e-12

/** P(|X̄ − p| ≥ ε) for the mean of n Bernoulli(p) draws, summed exactly from the binomial pmf. */
function exactTail(n: number, p: number, eps: number): number {
  let total = 0
  for (let k = 0; k <= n; k++) {
    // A small tolerance keeps k/n exactly at p ± ε inside the event despite rounding.
    if (Math.abs(k / n - p) >= eps - 1e-12) {
      total += Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))
    }
  }
  return total
}

/** Exact two-sided tail of a Bernoulli mean against the Chebyshev and Hoeffding bounds, on a log scale. */
export function BoundsVsN() {
  const eps = useParam(0.1, { min: 0.02, max: 0.3, step: 0.01 })
  const p = useParam(0.5, { min: 0.05, max: 0.95, step: 0.05 })

  const series = useMemo((): XYSeries[] => {
    const chebyshev = NS.map((n) => Math.min(1, (p.value * (1 - p.value)) / (n * eps.value ** 2)))
    const hoeffding = NS.map((n) => Math.min(1, 2 * Math.exp(-2 * n * eps.value ** 2)))
    const exact = NS.map((n) => Math.max(FLOOR, exactTail(n, p.value, eps.value)))
    return [
      { name: 'exact', type: 'line', x: NS, y: exact, slot: 0 },
      { name: 'Chebyshev p(1−p)/(nε²)', type: 'line', x: NS, y: chebyshev.map((v) => Math.max(FLOOR, v)), slot: 1 },
      { name: 'Hoeffding 2e^(−2nε²)', type: 'line', x: NS, y: hoeffding.map((v) => Math.max(FLOOR, v)), slot: 2 },
    ]
  }, [eps.value, p.value])

  // Smallest n (from the two formulas) that brings each bound down to 0.05.
  const nCheb = Math.ceil((p.value * (1 - p.value)) / (0.05 * eps.value ** 2))
  const nHoeff = Math.ceil(Math.log(2 / 0.05) / (2 * eps.value ** 2))

  return (
    <Interactive
      title="Chebyshev and Hoeffding against the exact tail"
      caption="The probability that the mean of n Bernoulli(p) draws misses p by at least ε. The Chebyshev bound falls like 1/n, the Hoeffding bound and the exact tail fall exponentially. Hoeffding ignores the variance, so it is loosest when p is near 0 or 1, where Chebyshev can win for small n."
      controls={
        <>
          <ParamSlider label="deviation ε" param={eps} />
          <ParamSlider label="success probability p" param={p} />
        </>
      }
      readout={
        <>
          <Readout label="n for Chebyshev ≤ 0.05" value={formatNumber(nCheb)} />
          <Readout label="n for Hoeffding ≤ 0.05" value={formatNumber(nHoeff)} />
        </>
      }
    >
      <XYChart height={320} xLabel="n" yLabel="P(|X̄ − p| ≥ ε)" series={series} yLog yRange={[FLOOR, 1]} />
    </Interactive>
  )
}

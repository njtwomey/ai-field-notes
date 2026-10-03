import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'
import { binomialPmf } from '@/lib/math/tests'

/**
 * Two-sided binomial test of H₀: p = p₀ for s successes in n trials. The exact p-value sums the probabilities of all
 * outcomes no more likely than s; the normal approximation is shown with and without the continuity correction.
 */
export function ExactVsNormal() {
  const [n, setN] = useState(20)
  const [successes, setSuccesses] = useState(15)
  const [p0, setP0] = useState(0.5)
  const s = Math.min(successes, n)

  const result = useMemo(() => {
    const ks = Array.from({ length: n + 1 }, (_, k) => k)
    const pmf = ks.map((k) => binomialPmf(k, n, p0))
    // A small tolerance keeps outcomes with equal probability (by symmetry) on the same side.
    const extreme = ks.filter((k) => pmf[k] <= pmf[s] * (1 + 1e-9))
    const rest = ks.filter((k) => pmf[k] > pmf[s] * (1 + 1e-9))
    const exact = Math.min(
      1,
      extreme.reduce((a, k) => a + pmf[k], 0),
    )
    const mean = n * p0
    const sd = Math.sqrt(n * p0 * (1 - p0))
    const gap = Math.abs(s - mean)
    const plain = 2 * (1 - normalCdf(gap / sd))
    const corrected = Math.min(1, 2 * (1 - normalCdf(Math.max(gap - 0.5, 0) / sd)))
    const xs = linspace(-0.5, n + 0.5, 300)
    const series: XYSeries[] = [
      { name: 'less extreme than s', type: 'bar', x: rest, y: rest.map((k) => pmf[k]), muted: true },
      { name: 'at least as extreme as s', type: 'bar', x: extreme, y: extreme.map((k) => pmf[k]), slot: 0 },
      { name: 'normal approximation', type: 'line', x: xs, y: xs.map((x) => normalPdf((x - mean) / sd) / sd), slot: 1 },
    ]
    return { series, exact, plain, corrected }
  }, [n, s, p0])

  return (
    <Interactive
      title="Exact binomial p-value and its normal approximation"
      caption="Bars are the binomial distribution of the number of successes under H₀. The exact two-sided p-value adds up the coloured bars, the outcomes no more likely than the one observed. The normal curve approximates the bars; shifting the cut-off by half a bar, the continuity correction, matches the area of the bars much better."
      controls={
        <>
          <ParamSlider label="trials n" value={n} onChange={setN} min={5} max={100} step={1} />
          <ParamSlider label="successes s" value={s} onChange={setSuccesses} min={0} max={n} step={1} />
          <ParamSlider label="null proportion p₀" value={p0} onChange={setP0} min={0.05} max={0.95} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="exact p" value={formatNumber(result.exact)} />
          <Readout label="normal p" value={formatNumber(result.plain)} />
          <Readout label="normal p, continuity-corrected" value={formatNumber(result.corrected)} />
        </>
      }
    >
      <XYChart height={280} series={result.series} xLabel="successes" yLabel="probability" yRange={[0, undefined]} />
    </Interactive>
  )
}

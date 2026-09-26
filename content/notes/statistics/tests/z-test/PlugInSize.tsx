import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { normalQuantile, studentTCdf } from '@/lib/math/special'

const N_MAX = 60

/**
 * The real Type I error rate of a two-sided "z-test" that plugs the sample standard deviation into the z statistic.
 * With normal data the statistic is exactly t with n − 1 degrees of freedom, so its size is P(|T| > z₁₋α/₂).
 */
export function PlugInSize() {
  const [alpha, setAlpha] = useState(0.05)
  const [n, setN] = useState(10)

  const result = useMemo(() => {
    const crit = normalQuantile(1 - alpha / 2)
    const size = (m: number) => 2 * (1 - studentTCdf(crit, m - 1))
    const ns = Array.from({ length: N_MAX - 1 }, (_, i) => i + 2)
    const series: XYSeries[] = [
      { name: 'z-test with s: actual size', type: 'line', x: ns, y: ns.map(size), slot: 0 },
      { name: 't-test: size α', type: 'line', x: [2, N_MAX], y: [alpha, alpha], slot: 1, dashed: true },
    ]
    return { series, size }
  }, [alpha])

  return (
    <Interactive
      title="Plugging s into a z-test"
      caption="The z-test assumes the population standard deviation σ is known. Replacing it with the sample standard deviation s and still using the normal critical value rejects a true null too often, because s is itself uncertain. The t-test uses the t distribution and has exactly size α for normal data."
      controls={
        <>
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.1}
            step={0.005}
          />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={2} max={N_MAX} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="actual size at this n" value={formatNumber(result.size(n))} />
          <Readout label="inflation" value={`${formatNumber(result.size(n) / alpha)}×`} />
        </>
      }
    >
      <XYChart
        height={280}
        series={result.series}
        xLabel="sample size n"
        yLabel="P(reject | H₀)"
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}

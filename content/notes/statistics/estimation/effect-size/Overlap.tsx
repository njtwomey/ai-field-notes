import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'

/** Two unit-variance normal groups whose means differ by d, with three readings of the same d. */
export function Overlap() {
  const [d, setD] = useState(0.5)

  const series = useMemo((): XYSeries[] => {
    const xs = linspace(-4, 4 + d, 300)
    const shared = xs.map((x) => Math.min(normalPdf(x), normalPdf(x - d)))
    return [
      { name: 'control', type: 'line', x: xs, y: xs.map((x) => normalPdf(x)), slot: 0 },
      { name: 'treatment', type: 'line', x: xs, y: xs.map((x) => normalPdf(x - d)), slot: 1 },
      { name: 'overlap', type: 'line', x: xs, y: shared, slot: 2, area: true },
    ]
  }, [d])

  return (
    <Interactive
      title="One effect size, three readings"
      caption="Two groups with equal standard deviations whose means differ by d standard deviations. The shaded area is the overlap of the two distributions. Cohen's U₃ is the share of the treatment group above the control mean. The probability of superiority is the chance that a random treated unit exceeds a random control unit."
      controls={<ParamSlider label="Cohen's d" value={d} onChange={setD} min={0} max={3} step={0.05} />}
      readout={
        <>
          <Readout label="overlap 2Φ(−d/2)" value={`${formatNumber(100 * 2 * normalCdf(-d / 2))}%`} />
          <Readout label="U₃ = Φ(d)" value={`${formatNumber(100 * normalCdf(d))}%`} />
          <Readout label="P(treated > control) = Φ(d/√2)" value={formatNumber(normalCdf(d / Math.SQRT2))} />
        </>
      }
    >
      <XYChart
        height={280}
        series={series}
        xLabel="outcome (control standard deviations)"
        yLabel="density"
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}

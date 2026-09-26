import { useMemo, useState } from 'react'
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
import { tPower } from '@/lib/math/tests'

/**
 * Power against a mean difference δ (in units of the outcome's standard deviation σ) for two designs with the same
 * number of measurements per condition: n pairs measured twice, with within-pair correlation ρ, or two independent
 * groups of n. The paired difference has standard deviation σ√(2(1 − ρ)); the independent difference σ√(2/n).
 */
export function PairingPower() {
  const [n, setN] = useState(15)
  const [effect, setEffect] = useState(0.5)
  const [alpha, setAlpha] = useState(0.05)
  const rho = useParam(0.6, { min: -0.5, max: 0.95, step: 0.01 })

  const result = useMemo(() => {
    const paired = (r: number) => tPower((effect * Math.sqrt(n)) / Math.sqrt(2 * (1 - r)), n - 1, alpha)
    const independent = tPower(effect / Math.sqrt(2 / n), 2 * n - 2, alpha)
    const rs = linspace(-0.5, 0.95, 146)
    const series: XYSeries[] = [
      { name: 'paired design', type: 'line', x: rs, y: rs.map(paired), slot: 0 },
      {
        name: 'two independent groups',
        type: 'line',
        x: [-0.5, 0.95],
        y: [independent, independent],
        slot: 1,
        dashed: true,
      },
    ]
    return { series, paired, independent }
  }, [n, effect, alpha])

  const handles: Handle[] = [{ kind: 'x', at: rho.value, label: 'ρ', onDrag: rho.set }]

  return (
    <Interactive
      title="Pairing pays when measurements correlate"
      caption="Power of a two-sided t-test to detect a mean difference, for n pairs measured under both conditions against two independent groups of n. The paired design removes the variation shared within a pair, so its power grows with the correlation ρ between the two measurements. Below ρ = 0 pairing hurts. Drag the line labelled ρ, or use its slider."
      controls={
        <>
          <ParamSlider label="within-pair correlation ρ" param={rho} />
          <ParamSlider label="pairs n" value={n} onChange={setN} min={3} max={100} step={1} />
          <ParamSlider
            label="true difference δ/σ"
            value={effect}
            onChange={setEffect}
            min={0.05}
            max={1.5}
            step={0.05}
          />
          <ParamSlider
            label="significance level α"
            value={alpha}
            onChange={setAlpha}
            min={0.01}
            max={0.1}
            step={0.005}
          />
        </>
      }
      readout={
        <>
          <Readout label="power, paired" value={formatNumber(result.paired(rho.value))} />
          <Readout label="power, independent" value={formatNumber(result.independent)} />
          <Readout label="variance ratio 1 − ρ" value={formatNumber(1 - rho.value)} />
        </>
      }
    >
      <XYChart
        height={280}
        series={result.series}
        xRange={[-0.5, 0.95]}
        yRange={[0, 1]}
        xLabel="within-pair correlation ρ"
        yLabel="power"
        handles={handles}
      />
    </Interactive>
  )
}

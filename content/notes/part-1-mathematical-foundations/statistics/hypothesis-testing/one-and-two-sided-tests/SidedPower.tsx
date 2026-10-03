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
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { normalCdf, normalQuantile } from '@/lib/math/special'

/** Power of the one-sided (H₁: μ > μ₀) and two-sided z-tests as functions of the true standardised effect δ. */
export function SidedPower() {
  const [n, setN] = useState(25)
  const [alpha, setAlpha] = useState(0.05)
  const effect = useParam(0.3, { min: -1, max: 1, step: 0.01 })

  const result = useMemo(() => {
    const one = normalQuantile(1 - alpha)
    const two = normalQuantile(1 - alpha / 2)
    const oneSided = (d: number) => 1 - normalCdf(one - d * Math.sqrt(n))
    const twoSided = (d: number) => {
      const s = d * Math.sqrt(n)
      return 1 - normalCdf(two - s) + normalCdf(-two - s)
    }
    const ds = linspace(-1, 1, 201)
    const series: XYSeries[] = [
      { name: 'one-sided, H₁: μ > μ₀', type: 'line', x: ds, y: ds.map(oneSided), slot: 0 },
      { name: 'two-sided, H₁: μ ≠ μ₀', type: 'line', x: ds, y: ds.map(twoSided), slot: 1 },
      { name: 'α', type: 'line', x: [-1, 1], y: [alpha, alpha], slot: 2, dashed: true },
    ]
    return { series, one, two, oneSided, twoSided }
  }, [n, alpha])

  const handles: Handle[] = [{ kind: 'x', at: effect.value, label: 'δ', onDrag: effect.set }]

  return (
    <Interactive
      title="Where each test spends its α"
      caption="Power of a z-test against a true standardised effect δ. The one-sided test is more powerful for δ > 0 and has almost no power for δ < 0. The two-sided test is symmetric. Both reject a true null (δ = 0) with probability α. Drag the line labelled δ, or use its slider, to read both powers."
      controls={
        <>
          <ParamSlider label="true effect δ (standard deviations)" param={effect} />
          <ParamSlider label="observations n" value={n} onChange={setN} min={5} max={200} step={1} />
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
          <Readout label="critical z, one-sided" value={formatNumber(result.one)} />
          <Readout label="critical |z|, two-sided" value={formatNumber(result.two)} />
          <Readout label="power, one-sided" value={formatNumber(result.oneSided(effect.value))} />
          <Readout label="power, two-sided" value={formatNumber(result.twoSided(effect.value))} />
        </>
      }
    >
      <XYChart
        height={300}
        series={result.series}
        xRange={[-1, 1]}
        yRange={[0, 1]}
        xLabel="true effect δ"
        yLabel="P(reject H₀)"
        handles={handles}
      />
    </Interactive>
  )
}

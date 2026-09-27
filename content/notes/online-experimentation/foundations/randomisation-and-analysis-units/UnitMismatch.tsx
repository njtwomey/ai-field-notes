import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { normalCdf, normalQuantile } from '@/lib/math/special'

const M_MAX = 50
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]

/** Probability that a test which ignores within-unit correlation rejects a true null: |N(0, D)| ≥ z_{1−α/2}. */
function falsePositiveRate(m: number, rho: number, alpha: number): number {
  const deff = 1 + (m - 1) * rho
  return 2 * (1 - normalCdf(normalQuantile(1 - alpha / 2) / Math.sqrt(deff)))
}

/**
 * A/A tests randomised by user but analysed as if every page view were independent. With m views per user and
 * intra-user correlation ρ, the true variance of the mean is 1 + (m − 1)ρ times the naive one, so the naive z-statistic
 * is N(0, D) under the null rather than N(0, 1).
 */
export function UnitMismatch() {
  const [rho, setRho] = useState(0.1)
  const [alpha, setAlpha] = useState('0.05')
  const m = useParam(10, { min: 1, max: M_MAX, step: 1 })

  const series = useMemo<XYSeries[]>(() => {
    const a = Number(alpha)
    const ms = Array.from({ length: M_MAX }, (_, i) => i + 1)
    return [
      { name: 'false-positive rate', type: 'line', x: ms, y: ms.map((k) => falsePositiveRate(k, rho, a)), slot: 0 },
      { name: 'α', type: 'line', x: [1, M_MAX], y: [a, a], slot: 1, dashed: true },
    ]
  }, [rho, alpha])

  const deff = 1 + (m.value - 1) * rho
  const handles: Handle[] = [{ kind: 'x', at: m.value, label: 'm', onDrag: (x) => m.set(Math.round(x)) }]

  return (
    <Interactive
      title="A/A false positives when the analysis unit is finer than the randomisation unit"
      caption="Users are randomised, but the test treats each of a user's m page views as an independent observation. Views of one user are correlated with intra-user correlation ρ, so the naive standard error is too small by a factor √(1 + (m − 1)ρ). The curve is the resulting probability that an A/A test is declared significant. Drag the line labelled m, or use its slider, to change the number of views per user."
      controls={
        <>
          <ParamSlider label="page views per user, m" param={m} />
          <ParamSlider label="intra-user correlation ρ" value={rho} onChange={setRho} min={0} max={0.5} step={0.01} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
        </>
      }
      readout={
        <>
          <Readout label="design effect 1 + (m − 1)ρ" value={formatNumber(deff)} />
          <Readout label="naive se ÷ true se" value={formatNumber(1 / Math.sqrt(deff))} />
          <Readout
            label="A/A false-positive rate"
            value={formatNumber(falsePositiveRate(m.value, rho, Number(alpha)))}
          />
        </>
      }
    >
      <XYChart
        height={300}
        series={series}
        xRange={[1, M_MAX]}
        yRange={[0, undefined]}
        xLabel="page views per user, m"
        yLabel="P(significant A/A test)"
        handles={handles}
      />
    </Interactive>
  )
}

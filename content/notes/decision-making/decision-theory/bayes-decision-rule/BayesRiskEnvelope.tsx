import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

const P = linspace(0, 1, 201)

/**
 * Conditional risk of each action as a function of the posterior p = P(y = 1 | x) for a binary problem: predicting
 * negative costs p·c_FN, predicting positive costs (1 − p)·c_FP, and deferring costs d whatever the class. The Bayes
 * rule takes the lowest line, so the Bayes risk is their lower envelope, a concave function of p.
 */
export function BayesRiskEnvelope() {
  const cfp = useParam(1, { min: 0.1, max: 5, step: 0.1 })
  const cfn = useParam(4, { min: 0.1, max: 5, step: 0.1 })
  const d = useParam(0.6, { min: 0.05, max: 3, step: 0.05 })
  const p = useParam(0.3, { min: 0, max: 1, step: 0.005 })
  const [defer, setDefer] = useState(true)

  const r = useMemo(() => {
    const neg = P.map((q) => q * cfn.value)
    const pos = P.map((q) => (1 - q) * cfp.value)
    const def = P.map(() => d.value)
    const env = P.map((_, i) => Math.min(neg[i], pos[i], defer ? def[i] : Infinity))
    return { neg, pos, def, env }
  }, [cfp.value, cfn.value, d.value, defer])

  const risks = {
    'predict negative': p.value * cfn.value,
    'predict positive': (1 - p.value) * cfp.value,
    ...(defer ? { defer: d.value } : {}),
  }
  const best = Object.entries(risks).reduce((a, b) => (b[1] < a[1] ? b : a))
  const threshold = cfp.value / (cfp.value + cfn.value)
  const top = Math.max(cfp.value, cfn.value) * 1.05

  const series: XYSeries[] = [
    { name: 'predict negative: p·c_FN', type: 'line', x: P, y: r.neg, slot: 0 },
    { name: 'predict positive: (1 − p)·c_FP', type: 'line', x: P, y: r.pos, slot: 1 },
    ...(defer ? [{ name: 'defer: d', type: 'line' as const, x: P, y: r.def, slot: 2 }] : []),
    { name: 'Bayes risk (lower envelope)', type: 'line', x: P, y: r.env, emphasis: true },
    { name: 'this case', type: 'scatter', x: [p.value], y: [best[1]], emphasis: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'posterior p', onDrag: (x) => p.set(x) }]

  return (
    <Interactive
      title="The Bayes rule takes the lowest expected loss"
      caption="Each line is the expected loss of one action for a case whose posterior probability of being positive is p. The Bayes rule picks the lowest line, so the minimum expected loss (thick line) is the lower envelope of straight lines and therefore concave in p. Without the defer action, the two error lines cross at p* = c_FP / (c_FP + c_FN). Drag the vertical line to move the case's posterior; change the costs and watch the crossings move."
      controls={
        <>
          <ParamSlider label="false-positive cost c_FP" param={cfp} />
          <ParamSlider label="false-negative cost c_FN" param={cfn} />
          <ParamSlider label="cost of deferring d" param={d} />
          <ParamSwitch label="allow deferring" checked={defer} onChange={setDefer} />
        </>
      }
      readout={
        <>
          <Readout label="p" value={formatNumber(p.value)} />
          <Readout label="Bayes action" value={best[0]} />
          <Readout label="its expected loss" value={formatNumber(best[1])} />
          <Readout label="p* without deferring" value={formatNumber(threshold)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="posterior p = P(y = 1 | x)"
        yLabel="expected loss of the action"
        xRange={[0, 1]}
        yRange={[0, top]}
        series={series}
        handles={handles}
      />
    </Interactive>
  )
}

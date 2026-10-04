import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const P = toFlat(linspace(0, 1, 201))

/**
 * Conditional risk of each action as a function of the posterior p = P(y = 1 | x) for a binary problem: predicting
 * negative costs p·c_FN, predicting positive costs (1 − p)·c_FP, and deferring costs d whatever the class. The Bayes
 * rule takes the lowest line, so the Bayes risk is their lower envelope, a concave function of p.
 */
export function BayesRiskEnvelope() {
  const state = useFigureState({
    cfp: float(1, { min: 0.1, max: 5, step: 0.1, label: 'false-positive cost c_FP' }),
    cfn: float(4, { min: 0.1, max: 5, step: 0.1, label: 'false-negative cost c_FN' }),
    d: float(0.6, { min: 0.05, max: 3, step: 0.05, label: 'cost of deferring d' }),
    defer: setting(true, 'allow deferring'),
    p: slider(0, 1, 0.3, { step: 0.005, onChart: true }),
  })

  const r = useMemo(() => {
    const neg = P.map((q) => q * state.cfn)
    const pos = P.map((q) => (1 - q) * state.cfp)
    const def = P.map(() => state.d)
    const env = P.map((_, i) => Math.min(neg[i], pos[i], state.defer ? def[i] : Infinity))
    return { neg, pos, def, env }
  }, [state.cfp, state.cfn, state.d, state.defer])

  const risks = {
    'predict negative': state.p * state.cfn,
    'predict positive': (1 - state.p) * state.cfp,
    ...(state.defer ? { defer: state.d } : {}),
  }
  const best = Object.entries(risks).reduce((a, b) => (b[1] < a[1] ? b : a))
  const threshold = state.cfp / (state.cfp + state.cfn)
  const top = Math.max(state.cfp, state.cfn) * 1.05

  const series: SeriesSpec[] = [
    { name: 'predict negative: p·c_FN', type: 'line', x: P, y: r.neg, slot: 0 },
    { name: 'predict positive: (1 − p)·c_FP', type: 'line', x: P, y: r.pos, slot: 1 },
    ...(state.defer ? [{ name: 'defer: d', type: 'line' as const, x: P, y: r.def, slot: 2 }] : []),
    { name: 'Bayes risk (lower envelope)', type: 'line', x: P, y: r.env, emphasis: true },
    { name: 'this case', type: 'scatter', x: [state.p], y: [best[1]], emphasis: true },
  ]

  const xAxis = useAxis({ label: 'posterior p = P(y = 1 | x)', range: [0, 1] })
  const yAxis = useAxis({ label: 'expected loss of the action', range: [0, top] })
  return (
    <Figure
      title="The Bayes rule takes the lowest expected loss"
      state={state}
      caption="Each line is the expected loss of one action for a case whose posterior probability of being positive is p. The Bayes rule picks the lowest line, so the minimum expected loss (thick line) is the lower envelope of straight lines and therefore concave in p. Without the defer action, the two error lines cross at p* = c_FP / (c_FP + c_FN). Drag the vertical line to move the case's posterior; change the costs and watch the crossings move."

      readouts={
        <>
          <Readout label="p" value={formatNumber(state.p)} />
          <Readout label="Bayes action" value={best[0]} />
          <Readout label="its expected loss" value={formatNumber(best[1])} />
          <Readout label="p* without deferring" value={formatNumber(threshold)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
        <Handle {...state.handle('p', { label: 'posterior p' })} />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 4000
const THRESHOLDS = toFlat(linspace(0.001, 0.999, 250))

/**
 * Expected cost per decision against the decision threshold, for a calibrated classifier on simulated data. Correct
 * decisions cost nothing (c₀₀ = c₁₁ = 0), so the optimal threshold is p* = c₁₀ / (c₁₀ + c₀₁).
 */
export function CostCurve() {
  const state = useFigureState({
    fp: int(5, { min: 1, max: 50, step: 1, label: 'false-positive cost c₁₀' }),
    fn: int(100, { min: 1, max: 200, step: 1, label: 'false-negative cost c₀₁' }),
    base: float(0.05, { min: 0.01, max: 0.5, step: 0.01, label: 'base rate P(j = 1)' }),
    separation: float(2.5, { min: 0.5, max: 5, step: 0.1, label: 'class separation' }),
    threshold: slider(0.001, 0.999, 0.5, { step: 0.001, onChart: true }),
  })

  // Simulated examples: a latent score is N(±d/2, 1) by class, and the probability reported for each example is the
  // exact posterior P(j = 1 | score) under that model, so the classifier is calibrated by construction.
  const data = useMemo(() => {
    const g = stream(7)
    const d = state.separation
    const pi = state.base
    const labels: boolean[] = []
    const probs: number[] = []
    for (let i = 0; i < N; i++) {
      const positive = uniform(g) < pi
      const s = (positive ? d / 2 : -d / 2) + normal(g)
      // Likelihood ratio of N(d/2, 1) to N(−d/2, 1) at s is exp(d s).
      const odds = (pi / (1 - pi)) * Math.exp(d * s)
      labels.push(positive)
      probs.push(odds / (1 + odds))
    }
    return { labels, probs }
  }, [state.base, state.separation])

  const pStar = state.fp / (state.fp + state.fn)

  const evaluate = useMemo(
    () => (t: number) => {
      let tp = 0
      let fpCount = 0
      let fnCount = 0
      let tn = 0
      for (let i = 0; i < N; i++) {
        const flag = data.probs[i] >= t
        if (data.labels[i]) {
          if (flag) tp++
          else fnCount++
        } else if (flag) fpCount++
        else tn++
      }
      return { tp, fp: fpCount, fn: fnCount, tn, cost: (fpCount * state.fp + fnCount * state.fn) / N }
    },
    [data, state.fp, state.fn],
  )

  const curve = useMemo(() => THRESHOLDS.map((t) => evaluate(t).cost), [evaluate])
  const top = Math.max(...curve) * 1.08
  const at = evaluate(state.threshold)
  const atStar = evaluate(pStar)
  const atHalf = evaluate(0.5)

  const xAxis = useAxis({ label: 'decision threshold on p', range: [0, 1] })
  const yAxis = useAxis({ label: 'expected cost per decision', range: [0, top] })
  return (
    <Figure
      title="Expected cost against the threshold"
      state={state}
      caption="Each simulated example gets a calibrated probability p of being positive and is flagged when p is at least the threshold. The curve is the average cost per decision; correct decisions cost nothing. The dashed line is Elkan's optimal threshold p* = c₁₀ / (c₁₀ + c₀₁), which sits at the bottom of the curve whatever the base rate or the classifier's quality. Drag the threshold, or change the costs and watch p* move."

      readouts={
        <>
          <Readout label="p*" value={formatNumber(pStar)} />
          <Readout label="cost at threshold" value={formatNumber(at.cost)} />
          <Readout label="cost at p*" value={formatNumber(atStar.cost)} />
          <Readout label="cost at 0.5" value={formatNumber(atHalf.cost)} />
          <Readout label="TP, FP, FN, TN" value={`${at.tp}, ${at.fp}, ${at.fn}, ${at.tn}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve name="expected cost" x={THRESHOLDS} y={curve} slot={0} />
        <Curve name="optimal threshold p*" x={[pStar, pStar]} y={[0, top]} slot={1} dashed />
        <Handle {...state.handle('threshold', { label: 'threshold' })} />
      </Plot>
    </Figure>
  )
}

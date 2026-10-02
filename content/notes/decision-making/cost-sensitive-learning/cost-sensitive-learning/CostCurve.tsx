import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Handle } from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const N = 4000
const THRESHOLDS = linspace(0.001, 0.999, 250)

/**
 * Expected cost per decision against the decision threshold, for a calibrated classifier on simulated data. Correct
 * decisions cost nothing (c₀₀ = c₁₁ = 0), so the optimal threshold is p* = c₁₀ / (c₁₀ + c₀₁).
 */
export function CostCurve() {
  const fp = useParam(5, { min: 1, max: 50, step: 1 })
  const fn = useParam(100, { min: 1, max: 200, step: 1 })
  const base = useParam(0.05, { min: 0.01, max: 0.5, step: 0.01 })
  const separation = useParam(2.5, { min: 0.5, max: 5, step: 0.1 })
  const threshold = useParam(0.5, { min: 0.001, max: 0.999, step: 0.001 })

  // Simulated examples: a latent score is N(±d/2, 1) by class, and the probability reported for each example is the
  // exact posterior P(j = 1 | score) under that model, so the classifier is calibrated by construction.
  const data = useMemo(() => {
    const g = rng(7)
    const d = separation.value
    const pi = base.value
    const labels: boolean[] = []
    const probs: number[] = []
    for (let i = 0; i < N; i++) {
      const positive = g.uniform() < pi
      const s = (positive ? d / 2 : -d / 2) + g.normal()
      // Likelihood ratio of N(d/2, 1) to N(−d/2, 1) at s is exp(d s).
      const odds = (pi / (1 - pi)) * Math.exp(d * s)
      labels.push(positive)
      probs.push(odds / (1 + odds))
    }
    return { labels, probs }
  }, [base.value, separation.value])

  const pStar = fp.value / (fp.value + fn.value)

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
      return { tp, fp: fpCount, fn: fnCount, tn, cost: (fpCount * fp.value + fnCount * fn.value) / N }
    },
    [data, fp.value, fn.value],
  )

  const curve = useMemo(() => THRESHOLDS.map((t) => evaluate(t).cost), [evaluate])
  const top = Math.max(...curve) * 1.08
  const at = evaluate(threshold.value)
  const atStar = evaluate(pStar)
  const atHalf = evaluate(0.5)

  const handles: Handle[] = [{ kind: 'x', at: threshold.value, label: 'threshold', onDrag: (x) => threshold.set(x) }]

  return (
    <Interactive
      title="Expected cost against the threshold"
      caption="Each simulated example gets a calibrated probability p of being positive and is flagged when p is at least the threshold. The curve is the average cost per decision; correct decisions cost nothing. The dashed line is Elkan's optimal threshold p* = c₁₀ / (c₁₀ + c₀₁), which sits at the bottom of the curve whatever the base rate or the classifier's quality. Drag the threshold, or change the costs and watch p* move."
      controls={
        <>
          <ParamSlider label="false-positive cost c₁₀" param={fp} />
          <ParamSlider label="false-negative cost c₀₁" param={fn} />
          <ParamSlider label="base rate P(j = 1)" param={base} />
          <ParamSlider label="class separation" param={separation} />
        </>
      }
      readout={
        <>
          <Readout label="p*" value={formatNumber(pStar)} />
          <Readout label="cost at threshold" value={formatNumber(at.cost)} />
          <Readout label="cost at p*" value={formatNumber(atStar.cost)} />
          <Readout label="cost at 0.5" value={formatNumber(atHalf.cost)} />
          <Readout label="TP, FP, FN, TN" value={`${at.tp}, ${at.fp}, ${at.fn}, ${at.tn}`} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="decision threshold on p"
        yLabel="expected cost per decision"
        xRange={[0, 1]}
        yRange={[0, top]}
        handles={handles}
        series={[
          { name: 'expected cost', type: 'line', x: THRESHOLDS, y: curve, slot: 0 },
          { name: 'optimal threshold p*', type: 'line', x: [pStar, pStar], y: [0, top], slot: 1, dashed: true },
        ]}
      />
    </Interactive>
  )
}

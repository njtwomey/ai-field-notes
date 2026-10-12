import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ModelGeneratedLabelsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'pseudo_threshold' | 'anchoring_bias' | 'soft_temperature'>(
      [
        { value: 'pseudo_threshold', label: 'Confidence Thresholding & Confirmation Bias (Lee 2013)' },
        { value: 'anchoring_bias', label: 'Human-in-the-Loop Anchoring Distortion (Tversky 1974)' },
        { value: 'soft_temperature', label: 'Distillation Soft Labels & Dark Knowledge (Hinton 2015)' },
      ],
      'pseudo_threshold',
      { label: 'diagnostic mode' },
    ),
    threshold: slider(0.5, 0.98, 0.85, {
      step: 0.02,
      label: 'confidence acceptance threshold tau',
    }),
    temperature: slider(1.0, 8.0, 3.0, {
      step: 0.5,
      label: 'distillation temperature tau',
    }),
  })

  const { visualizerMode, threshold, temperature } = state

  const results = useMemo(() => {
    if (visualizerMode === 'pseudo_threshold') {
      const thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
      // Coverage decreases as threshold increases: e.g. from 95% down to 12%
      const coverage = thresholds.map((t) => 100 * Math.exp(-2.5 * (t - 0.5)))
      // Error rate in pseudo-labels decreases as threshold increases: from 28% down to 1.8%
      const errorRate = thresholds.map((t) => (30 * Math.pow(1 - t, 0.8)) / Math.pow(0.5, 0.8))

      // Current values at selected threshold
      const currCov = 100 * Math.exp(-2.5 * (threshold - 0.5))
      const currErr = (30 * Math.pow(1 - threshold, 0.8)) / Math.pow(0.5, 0.8)

      return {
        x: thresholds.map((t) => t * 100),
        y: coverage,
        baseline: errorRate,
        stat1: `Threshold tau = ${(threshold * 100).toFixed(0)}%`,
        stat2: `Accepted volume: ${currCov.toFixed(1)}% coverage`,
        stat3: `Label noise: ${currErr.toFixed(1)}% error rate`,
        warning:
          threshold < 0.75
            ? 'Severe confirmation bias: high pseudo-label noise locks in student errors'
            : threshold > 0.92
              ? 'Extreme threshold starvation: minimal coverage stalls self-training progression'
              : 'Optimal operating point: high precision with substantial training volume',
        axisXLabel: 'Model Confidence Threshold tau (%)',
        axisYLabel: 'Metric Percentage (%)',
        xRange: [48, 100] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'anchoring_bias') {
      // Model error rate: 5% to 40%
      const modelErrors = [5, 10, 15, 20, 25, 30, 35, 40]
      // Independent human error rate baseline: flat at ~4%
      const blankSlateHumanError = modelErrors.map(() => 4.0)
      // Anchored human error rate when reviewing pre-filled suggestions:
      // Humans uncritically approve incorrect suggestions (rubber-stamping ~45% of model errors)
      const anchoredHumanError = modelErrors.map((e) => 4.0 + 0.45 * e)

      return {
        x: modelErrors,
        y: anchoredHumanError,
        baseline: blankSlateHumanError,
        stat1: 'Blank-slate error: 4.0%',
        stat2: `Anchored error: ${(4.0 + 0.45 * 20).toFixed(1)}% (at 20% model error)`,
        stat3: 'Approval heuristic: 45% pass-through',
        warning: 'Pre-filled model suggestions induce rubber-stamping, corrupting human QA audits',
        axisXLabel: 'Drafting Model Error Rate (%)',
        axisYLabel: 'Effective Post-Review Error Rate (%)',
        xRange: [0, 45] as [number, number],
        yRange: [0, 26] as [number, number],
      }
    } else {
      // 5 classes: logits = [4.5, 2.8, 1.2, -0.5, -2.0]
      const classes = [1, 2, 3, 4, 5]
      const rawLogits = [4.5, 2.8, 1.2, -0.5, -2.0]
      const T = temperature

      // Softmax with temperature: exp(z_i / T) / sum
      const scaledLogits = rawLogits.map((z) => z / T)
      const maxL = Math.max(...scaledLogits)
      const exps = scaledLogits.map((z) => Math.exp(z - maxL))
      const sumExp = exps.reduce((a, b) => a + b, 0)
      const softProbs = exps.map((e) => (e / sumExp) * 100)

      // Hard target baseline (class 1 is 100%, others 0%)
      const hardTargets = [100, 0, 0, 0, 0]

      // Entropy calculation
      const entropy = -softProbs
        .map((p) => p / 100)
        .filter((p) => p > 1e-6)
        .reduce((sum, p) => sum + p * Math.log2(p), 0)

      return {
        x: classes,
        y: softProbs,
        baseline: hardTargets,
        stat1: `Temperature T = ${T.toFixed(1)}`,
        stat2: `Soft label entropy: ${entropy.toFixed(2)} bits`,
        stat3: `Secondary class weight: ${softProbs[1].toFixed(1)}%`,
        warning:
          T <= 1.5
            ? 'Low temperature: approximates hard one-hot labels, hiding dark knowledge'
            : T >= 6.0
              ? 'High temperature: flattens logits into near-uniform distribution'
              : 'Informative dark knowledge: cross-class similarities guide student gradients',
        axisXLabel: 'Class Index (1: True Category, 2-5: Competing Classes)',
        axisYLabel: 'Target Probability Density (%)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, threshold, temperature])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Model-Generated Supervision & Human Anchoring Inspector"
      state={state}
      hoverReadout={false}
      caption="Quantifying model-generated label dynamics: self-training confidence threshold trade-offs, human cognitive anchoring during pre-filled QA review, and temperature-scaled knowledge distillation."
      readouts={
        <>
          <Readout label="Operating State" value={results.stat1} />
          <Readout label="Yield / Accuracy" value={results.stat2} />
          <Readout label="Distortion Metric" value={results.stat3} />
          <Readout
            label="Methodological Notice"
            value={results.warning}
            color={
              (visualizerMode === 'pseudo_threshold' && threshold < 0.75) || visualizerMode === 'anchoring_bias'
                ? 'var(--red-600)'
                : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Model-Generated Label Profiles">
        <Curve
          name={
            visualizerMode === 'pseudo_threshold'
              ? 'Label Error Rate'
              : visualizerMode === 'anchoring_bias'
                ? 'Blank-Slate Human Error Baseline'
                : 'Hard One-Hot Ground Truth'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'pseudo_threshold'
              ? 'Retained Data Coverage'
              : visualizerMode === 'anchoring_bias'
                ? 'Anchored Review Error Rate'
                : 'Soft Teacher Target Probabilities'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points
          name="Sample Points"
          x={results.x}
          y={results.y}
          slot={visualizerMode === 'anchoring_bias' ? 1 : 0}
          size={7}
        />
      </Plot>
    </Figure>
  )
}

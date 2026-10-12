import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DelayedAndPartialLabelsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'delayed_feedback' | 'pu_correction' | 'multilabel_mask'>(
      [
        { value: 'delayed_feedback', label: 'Delayed Feedback Dynamics (Chapelle 2014)' },
        { value: 'pu_correction', label: 'Positive-Unlabelled Calibration (Elkan-Noto 2008)' },
        { value: 'multilabel_mask', label: 'Multi-Label Partial Masking vs Zero-Imputation' },
      ],
      'delayed_feedback',
      { label: 'investigation mode' },
    ),
    elapsedDays: slider(1, 30, 7, {
      step: 1,
      label: 'elapsed observation window (days)',
    }),
    labelFrequency: slider(0.1, 0.9, 0.35, {
      step: 0.05,
      label: 'PU label frequency c = P(s=1 | y=1)',
    }),
  })

  const { visualizerMode, elapsedDays, labelFrequency } = state

  const results = useMemo(() => {
    if (visualizerMode === 'delayed_feedback') {
      // Days from 0 to 30
      const days = [0.5, 1, 2, 3, 5, 7, 10, 14, 21, 28, 30]
      const trueBaseRate = 12.0 // 12% lifetime conversion
      const scaleParam = 5.0 // median delay parameter in days

      // Cumulative conversion CDF: 1 - exp(-(t / scaleParam)^0.85)
      const cumRates = days.map((d) => {
        const frac = 1 - Math.exp(-Math.pow(d / scaleParam, 0.85))
        return trueBaseRate * frac
      })

      const trueLine = days.map(() => trueBaseRate)

      // At the selected elapsedDays
      const currentFrac = 1 - Math.exp(-Math.pow(elapsedDays / scaleParam, 0.85))
      const observedRate = trueBaseRate * currentFrac
      const falseNegContamination = trueBaseRate - observedRate

      return {
        x: days,
        y: cumRates,
        baseline: trueLine,
        stat1: `${observedRate.toFixed(1)}% observed positive`,
        stat2: `${(currentFrac * 100).toFixed(1)}% matured (c = ${currentFrac.toFixed(2)})`,
        stat3: `${falseNegContamination.toFixed(1)}% latent positives hidden in zeros`,
        warning:
          elapsedDays < 5
            ? 'Severe right-censoring: recent data poisons model with false negatives'
            : elapsedDays >= 21
              ? 'Matured cohort: minimal right-censoring bias'
              : 'Moderate delay: requires delayed feedback survival modeling',
        axisXLabel: 'Elapsed Observation Days (t - t_click)',
        axisYLabel: 'Conversion Rate Prevalence (%)',
        xRange: [0, 31] as [number, number],
        yRange: [0, 15] as [number, number],
      }
    } else if (visualizerMode === 'pu_correction') {
      // Map apparent score P(s=1|x) from 0.05 to 0.90 to calibrated P(y=1|x) = P(s=1|x) / c
      const scores = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
      const c = labelFrequency

      const calibrated = scores.map((s) => Math.min(1.0, s / c) * 100)
      const apparent = scores.map((s) => s * 100)

      return {
        x: scores.map((s) => s * 100),
        y: calibrated,
        baseline: apparent,
        stat1: `Label propensity c = ${(c * 100).toFixed(0)}%`,
        stat2: `Calibration multiplier: ${(1 / c).toFixed(2)}x`,
        stat3: `Underestimation: ${((1 - c) * 100).toFixed(0)}% suppression`,
        warning:
          c < 0.25
            ? 'Sparse positive labels: naive models severely underestimate true risk'
            : 'Controlled PU setting: Elkan-Noto rescaling restores unbiased posteriors',
        axisXLabel: 'Naive Apparent Score P(s=1 | x) [%]',
        axisYLabel: 'Calibrated Posterior P(y=1 | x) [%]',
        xRange: [0, 60] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Multi-label partial masking: 8 classes with varying observation rates
      // Classes 1 to 8: comparing gradient magnitude under zero imputation vs masked loss
      const classes = [1, 2, 3, 4, 5, 6, 7, 8]
      // Missing rate per class
      const missingRates = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85, 0.95]
      // Under zero imputation, false negative gradient pushes legitimate logits downwards by missingRate * co-occurrence
      const zeroImpPenalty = missingRates.map((m) => m * 42.0)
      const maskedLossBaseline = classes.map(() => 0)

      return {
        x: classes,
        y: zeroImpPenalty,
        baseline: maskedLossBaseline,
        stat1: 'Masked loss penalty: 0.0% (unbiased)',
        stat2: `Max zero-imputation penalty: ${Math.max(...zeroImpPenalty).toFixed(1)}%`,
        stat3: 'Tri-state loss: +1, -1, masked',
        warning: 'Zero-imputing unannotated labels destroys precision on frequent concepts',
        axisXLabel: 'Label Class Index (Ranked by Missingness)',
        axisYLabel: 'Spurious Downward Gradient Push (%)',
        xRange: [0.5, 8.5] as [number, number],
        yRange: [0, 45] as [number, number],
      }
    }
  }, [visualizerMode, elapsedDays, labelFrequency])

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
      title="Delayed Feedback & Partial Supervision Diagnostics"
      state={state}
      hoverReadout={false}
      caption="Interactive exploration of delayed conversion dynamics, positive-unlabelled (PU) probability rescaling, and multi-label partial annotation masking."
      readouts={
        <>
          <Readout label="Observed State" value={results.stat1} />
          <Readout label="Calibration Metric" value={results.stat2} />
          <Readout label="Latent Incompleteness" value={results.stat3} />
          <Readout
            label="Operational Warning"
            value={results.warning}
            color={
              (visualizerMode === 'delayed_feedback' && elapsedDays < 5) ||
              (visualizerMode === 'pu_correction' && labelFrequency < 0.25)
                ? 'var(--red-600)'
                : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Supervision Dynamics">
        <Curve
          name={
            visualizerMode === 'delayed_feedback'
              ? 'True Asymptotic Prevalence'
              : visualizerMode === 'pu_correction'
                ? 'Uncalibrated Apparent Line'
                : 'Unbiased Masked Loss Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'delayed_feedback'
              ? 'Observed Conversion Rate'
              : visualizerMode === 'pu_correction'
                ? 'Calibrated True Posterior P(y=1|x)'
                : 'Zero-Imputation Gradient Distortion'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points
          name="Evaluation Points"
          x={results.x}
          y={results.y}
          slot={visualizerMode === 'delayed_feedback' && elapsedDays < 5 ? 1 : 0}
          size={7}
        />
      </Plot>
    </Figure>
  )
}

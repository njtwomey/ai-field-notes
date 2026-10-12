import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AnnotatorAggregationVisualizer() {
  const state = useFigureState({
    aggregationLens: choice<'dawid_skene' | 'soft_labels' | 'glad_curve'>(
      [
        { value: 'dawid_skene', label: 'Dawid-Skene Probabilistic Weighting vs Majority Vote' },
        { value: 'soft_labels', label: 'Soft Label Target Distribution (Preserving Ambiguity)' },
        { value: 'glad_curve', label: 'GLAD: Expertise (α) and Instance Difficulty (β)' },
      ],
      'dawid_skene',
      { label: 'aggregation mechanism' },
    ),
    spammerRate: slider(0.1, 0.8, 0.35, {
      step: 0.05,
      label: 'noisy crowd rater fraction in pool',
    }),
    difficulty: slider(0.2, 3.0, 1.5, {
      step: 0.2,
      label: 'instance intrinsic difficulty (beta)',
    }),
  })

  const { aggregationLens, spammerRate, difficulty } = state

  const results = useMemo(() => {
    if (aggregationLens === 'dawid_skene') {
      // 5 annotators: A1, A2 (Senior Experts: 95% acc), A3 (Moderate: 75% acc), A4, A5 (Noisy/Spam: 45% acc)
      const raters = [1, 2, 3, 4, 5]
      const individualAcc = [95.0, 94.0, 75.0, 50.0 - spammerRate * 20, 48.0 - spammerRate * 20]

      // Effective weight assigned by Dawid-Skene: log-odds of accuracy
      // w_m = log(p / (1 - p))
      const weights = individualAcc.map((acc) => {
        const p = Math.max(0.1, Math.min(0.99, acc / 100))
        return Math.max(0, Math.log(p / (1 - p)))
      })

      // Majority vote treats all weights as 1.0
      const majWeights = raters.map(() => 1.0)

      return {
        x: raters,
        y: weights,
        baseline: majWeights,
        status1: 'Experts (A1, A2) receive 3.5x higher voting weight than noisy workers',
        status2: spammerRate > 0.4 ? 'Majority vote vulnerable to spammer collusions' : 'Robust consensus achieved',
        alert: 'Dawid-Skene automatically suppresses low-accuracy raters without manual intervention',
      }
    } else if (aggregationLens === 'soft_labels') {
      // Cross-entropy loss profile as model predicted probability p ranges from 0.05 to 0.95
      // Ground truth human distribution is ambiguous: [p_true = 0.60, 1 - p_true = 0.40]
      const pGrid = [5, 15, 25, 35, 45, 55, 60, 65, 75, 85, 95]
      // Hard one-hot loss (forced label 1.0): -log(p)
      const hardLoss = pGrid.map((p) => -Math.log(p / 100))
      // Soft label loss: - (0.6 * log(p) + 0.4 * log(1 - p))
      const softLoss = pGrid.map((p) => {
        const prob = p / 100
        return -(0.6 * Math.log(prob) + 0.4 * Math.log(1 - prob))
      })

      return {
        x: pGrid,
        y: softLoss,
        baseline: hardLoss,
        status1: 'Minimum soft loss achieved at p = 0.60 (Perfect Calibration)',
        status2: 'Hard label forces model to p = 1.0 (Overconfident memorization)',
        alert: 'Soft labels encourage calibrated output probabilities matching human disagreement',
      }
    } else {
      // GLAD logistic curve: P(correct) = 1 / (1 + exp(-alpha * beta))
      // Sweep expertise alpha from -2 to +4 across the given difficulty beta
      const alphaGrid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
      const pCorrect = alphaGrid.map((a) => {
        const val = 1 / (1 + Math.exp(-a * difficulty))
        return val * 100
      })
      const randomLine = alphaGrid.map(() => 50.0)

      return {
        x: alphaGrid,
        y: pCorrect,
        baseline: randomLine,
        status1: `Difficulty β = ${difficulty.toFixed(1)}`,
        status2:
          difficulty > 2.0 ? 'Steep sigmoid: Expertise dominates performance' : 'Flat sigmoid: High inherent ambiguity',
        alert: 'GLAD decouples worker expertise from item-level difficulty',
      }
    }
  }, [aggregationLens, spammerRate, difficulty])

  const axisX = useAxis({
    label:
      aggregationLens === 'dawid_skene'
        ? 'Annotator Identifier (1-2: Senior Experts, 3: Average, 4-5: Noisy Crowd)'
        : aggregationLens === 'soft_labels'
          ? 'Model Predicted Probability for Class 1 (%)'
          : 'Annotator Expertise Score (α)',
    range: aggregationLens === 'dawid_skene' ? [0.5, 5.5] : aggregationLens === 'soft_labels' ? [0, 100] : [-2.2, 4.2],
  })

  const axisY = useAxis({
    label:
      aggregationLens === 'dawid_skene'
        ? 'Inferred Voting Weight w_m = ln(p / (1 - p))'
        : aggregationLens === 'soft_labels'
          ? 'Cross-Entropy Training Loss'
          : 'Probability of Correct Annotation (%)',
    range: aggregationLens === 'dawid_skene' ? [0, 4.0] : aggregationLens === 'soft_labels' ? [0.4, 3.2] : [0, 105],
  })

  return (
    <Figure
      title="Annotator Aggregation & Consensus Modeling Explorer"
      state={state}
      hoverReadout={false}
      caption="Comparing consensus aggregation strategies: Dawid-Skene down-weights noisy crowd raters using inferred log-odds weights, while soft distribution labels preserve natural human ambiguity to improve probability calibration."
      readouts={
        <>
          <Readout label="Primary Diagnostic" value={results.status1} />
          <Readout label="Aggregation Behavior" value={results.status2} />
          <Readout label="Theoretical Insight" value={results.alert} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Aggregation Diagnostic Profile">
        <Curve name="Dawid-Skene Weight / Soft Loss Profile" x={results.x} y={results.y} slot={0} width={2.5} />
        <Curve
          name="Unweighted Majority Vote / Hard One-Hot Loss"
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Points name="Operating Diagnostic Point" x={results.x} y={results.y} slot={1} size={7} />
      </Plot>
    </Figure>
  )
}

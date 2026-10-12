import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AugmentationPrinciplesVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'mixup_manifold' | 'magnitude_tradeoff' | 'consistency_training'>(
      [
        { value: 'mixup_manifold', label: 'Mixup Linear Interpolation & Boundary Smoothing (Zhang 2018)' },
        { value: 'magnitude_tradeoff', label: 'Augmentation Intensity vs Label Preservation Hazard' },
        { value: 'consistency_training', label: 'Weak-to-Strong Consistency Training (FixMatch)' },
      ],
      'mixup_manifold',
      { label: 'augmentation principle' },
    ),
    mixupAlpha: slider(0.1, 1.0, 0.4, {
      step: 0.1,
      label: 'Mixup Beta distribution parameter alpha',
    }),
    distortionMagnitude: slider(1, 10, 5, {
      step: 1,
      label: 'augmentation distortion intensity (1: Mild, 10: Extreme)',
    }),
  })

  const { visualizerMode, mixupAlpha, distortionMagnitude } = state

  const results = useMemo(() => {
    if (visualizerMode === 'mixup_manifold') {
      // Interpolation lambda from 0.0 to 1.0 (0: 100% Class B, 1: 100% Class A)
      const lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      // Ideal Mixup target is perfectly linear: P(Class A) = lambda
      const mixupLinear = lambdas.map((l) => l * 100)
      // Standard Empirical Risk Minimization (ERM) model: sharp step function / overconfident sigmoid
      const ermModel = lambdas.map((l) => {
        const z = 14 * (l - 0.5)
        return (1 / (1 + Math.exp(-z))) * 100
      })

      return {
        x: lambdas.map((l) => l * 100),
        y: mixupLinear,
        baseline: ermModel,
        stat1: `Mixup Beta(alpha=${mixupAlpha.toFixed(1)}, alpha=${mixupAlpha.toFixed(1)})`,
        stat2: 'Enforces linear inter-class manifold',
        stat3: 'Suppresses uncalibrated out-of-data overconfidence',
        warning: 'Mixup smooths decision boundaries, preventing neural networks from memorizing isolated points',
        axisXLabel: 'Interpolation Ratio lambda (% Class A Weight)',
        axisYLabel: 'Assigned Probability P(Class A) [%]',
        xRange: [0, 100] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'magnitude_tradeoff') {
      // Magnitudes 1 to 10
      const mags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Regularization robustness benefit: increases monotonically
      const robustness = mags.map((m) => 50 + 42 * (1 - Math.exp(-m / 2.5)))
      // Label preservation / semantic fidelity: drops sharply at high distortion
      const labelIntegrity = mags.map((m) => {
        if (m <= 4) return 100 - (m - 1) * 2.0
        return Math.max(10, 94 - Math.pow(m - 4, 2) * 2.2)
      })
      // Net validation accuracy: product of both factors
      const netAccuracy = mags.map((_, i) => (robustness[i] * labelIntegrity[i]) / 100)

      const currIntegrity = labelIntegrity[distortionMagnitude - 1]
      const currAcc = netAccuracy[distortionMagnitude - 1]

      return {
        x: mags,
        y: netAccuracy,
        baseline: labelIntegrity,
        stat1: `Intensity level: ${distortionMagnitude} / 10`,
        stat2: `Label semantic integrity: ${currIntegrity.toFixed(1)}%`,
        stat3: `Net validation accuracy: ${currAcc.toFixed(1)}%`,
        warning:
          distortionMagnitude >= 8
            ? 'CRITICAL HAZARD: Over-augmentation destroys semantic ground truth labels'
            : distortionMagnitude <= 2
              ? 'Mild augmentation: insufficient regularization leave model prone to overfitting'
              : 'Optimal operating window: high regularization with preserved label integrity',
        axisXLabel: 'Augmentation Distortion Intensity Level',
        axisYLabel: 'Performance & Integrity Metric (%)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Confidence thresholds tau: 0.60 to 0.98
      const thresholds = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
      // Retained unlabeled samples passing weak-view threshold
      const passRate = thresholds.map((t) => 100 * Math.exp(-2.2 * (t - 0.6)))
      // Consistency gradient quality (% uncontaminated)
      const gradQuality = thresholds.map((t) => 55 + 44 * Math.pow((t - 0.6) / 0.38, 0.7))

      return {
        x: thresholds.map((t) => t * 100),
        y: gradQuality,
        baseline: passRate,
        stat1: 'Weak view: standard crop & flip',
        stat2: 'Strong view: RandAugment + CutMix',
        stat3: 'Threshold tau = 95%: 94% gradient purity',
        warning: 'High confidence threshold on weak view prevents confirmation bias on strong view',
        axisXLabel: 'Weak-View Confidence Threshold tau (%)',
        axisYLabel: 'Metric Percentage (%)',
        xRange: [58, 100] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, mixupAlpha, distortionMagnitude])

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
      title="Data Augmentation Principles & Consistency Inspector"
      state={state}
      hoverReadout={false}
      caption="Evaluating core data augmentation mechanics: Mixup convex interpolation preventing overconfidence, the intensity versus label corruption trade-off, and semi-supervised consistency training."
      readouts={
        <>
          <Readout label="Operating Mode" value={results.stat1} />
          <Readout label="Primary Dynamic" value={results.stat2} />
          <Readout label="Regularization Metric" value={results.stat3} />
          <Readout
            label="Methodological Notice"
            value={results.warning}
            color={
              visualizerMode === 'magnitude_tradeoff' && distortionMagnitude >= 8 ? 'var(--red-600)' : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Augmentation Dynamic Trajectory">
        <Curve
          name={
            visualizerMode === 'mixup_manifold'
              ? 'Standard ERM Overconfident Model'
              : visualizerMode === 'magnitude_tradeoff'
                ? 'Label Semantic Integrity Boundary'
                : 'Unlabeled Passing Coverage'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'mixup_manifold'
              ? 'Mixup Linear Manifold Target'
              : visualizerMode === 'magnitude_tradeoff'
                ? 'Net Validation Performance'
                : 'Consistency Gradient Purity'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating States" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

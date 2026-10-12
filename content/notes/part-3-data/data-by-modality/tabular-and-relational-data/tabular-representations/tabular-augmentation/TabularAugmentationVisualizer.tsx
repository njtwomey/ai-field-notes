import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TabularAugmentationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'feature_swapping' | 'mixup_embedding' | 'noise_scaling'>(
      [
        { value: 'feature_swapping', label: 'Feature Swapping (SAINT / CutMix) vs Correlation Integrity' },
        { value: 'mixup_embedding', label: 'Input Space vs Embedding Space Mixup (Categorical Validity)' },
        { value: 'noise_scaling', label: 'Noise Injection: Unscaled Raw vs Robust IQR-Scaled Noise' },
      ],
      'feature_swapping',
      { label: 'augmentation strategy' },
    ),
    swapRate: slider(0.05, 0.5, 0.2, {
      step: 0.05,
      label: 'feature swap probability p_swap',
    }),
    noiseMagnitude: slider(0.05, 0.5, 0.15, {
      step: 0.05,
      label: 'Gaussian noise magnitude sigma',
    }),
  })

  const { visualizerMode, swapRate, noiseMagnitude } = state

  const results = useMemo(() => {
    if (visualizerMode === 'feature_swapping') {
      // Swap probability from 0.05 to 0.50
      const rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
      // Regularization benefit: peaks around 0.20-0.25
      const regBenefit = rates.map((r) => 50 + 40 * Math.sin((r / 0.5) * Math.PI))
      // Correlation violation hazard (% of synthetic rows with physically impossible combinations)
      const violationHazard = rates.map((r) => 100 * (1 - Math.pow(1 - r, 3.2)))

      const currBenefit = 50 + 40 * Math.sin((swapRate / 0.5) * Math.PI)
      const currHazard = 100 * (1 - Math.pow(1 - swapRate, 3.2))

      return {
        x: rates.map((r) => r * 100),
        y: regBenefit,
        baseline: violationHazard,
        stat1: `Swap rate: ${(swapRate * 100).toFixed(0)}% columns`,
        stat2: `Regularization gain: ${currBenefit.toFixed(1)}%`,
        stat3: `Joint violation hazard: ${currHazard.toFixed(1)}%`,
        warning:
          swapRate > 0.3
            ? 'High swap rate breaks joint tabular correlations (e.g. age vs career seniority mismatches)'
            : 'Moderate swap rate preserves marginal validity while regularizing deep tabular models',
        axisXLabel: 'Column Swapping Probability p_swap (%)',
        axisYLabel: 'Metric Percentage (%)',
        xRange: [0, 52] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'mixup_embedding') {
      // Lambda from 0 to 1
      const lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      // Post-embedding mixup representation validity: stays 100% valid continuous vector
      const embedValidity = lambdas.map(() => 100.0)
      // Input-space naive categorical mixup: completely invalid non-integer floats (validity 0% except at endpoints)
      const inputValidity = lambdas.map((l) => (l === 0.0 || l === 1.0 ? 100.0 : 0.0))

      return {
        x: lambdas.map((l) => l * 100),
        y: embedValidity,
        baseline: inputValidity,
        stat1: 'Embedding Mixup: 100% valid manifold',
        stat2: 'Input-space Mixup: 0% valid categories',
        stat3: 'Interpolates dense semantic tokens',
        warning: 'Never mixup raw categorical IDs in input space; interpolate in post-embedding token space',
        axisXLabel: 'Interpolation Weight lambda (% Row A Weight)',
        axisYLabel: 'Categorical Representation Semantic Validity (%)',
        xRange: [0, 100] as [number, number],
        yRange: [-5, 108] as [number, number],
      }
    } else {
      // 3 features with differing scales: Col 1 (Age ~ 40), Col 2 (Income ~ 75,000), Col 3 (Credit ~ 700)
      const cols = [1, 2, 3]
      // Unscaled noise SNR impact: destroys low-variance features while having 0 effect on high-variance features
      // Scaled robust IQR noise preserves balanced 15% SNR perturbation across all columns
      const iqrScaledSNR = cols.map(() => (1 - noiseMagnitude) * 100)
      const unscaledSNR = [15.0, 99.8, 88.0] // Col 1 destroyed (15%), Col 2 barely touched (99.8%)

      return {
        x: cols,
        y: iqrScaledSNR,
        baseline: unscaledSNR,
        stat1: `Noise magnitude sigma = ${noiseMagnitude.toFixed(2)}`,
        stat2: 'IQR-scaled: balanced across columns',
        stat3: 'Unscaled: 15% to 99.8% distortion skew',
        warning: 'Unscaled additive noise wrecks low-range columns while having zero effect on high-magnitude fields',
        axisXLabel: 'Feature Column (1: Age [0-100], 2: Income [$10k-$200k], 3: Credit [300-850])',
        axisYLabel: 'Signal-to-Noise Ratio Preservation (%)',
        xRange: [0.5, 3.5] as [number, number],
        yRange: [0, 108] as [number, number],
      }
    }
  }, [visualizerMode, swapRate, noiseMagnitude])

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
      title="Tabular Data Augmentation & Integrity Diagnostics"
      state={state}
      hoverReadout={false}
      caption="Evaluating tabular data augmentation mechanics: Feature swapping (SAINT) vs correlation breakdown, input-space versus embedding-space Mixup validity, and robust IQR noise scaling."
      readouts={
        <>
          <Readout label="Augmentation Setting" value={results.stat1} />
          <Readout label="Regularization Metric" value={results.stat2} />
          <Readout label="Integrity Impact" value={results.stat3} />
          <Readout
            label="Methodological Notice"
            value={results.warning}
            color={
              (visualizerMode === 'feature_swapping' && swapRate > 0.3) || visualizerMode === 'noise_scaling'
                ? 'var(--blue-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Tabular Augmentation Dynamics">
        <Curve
          name={
            visualizerMode === 'feature_swapping'
              ? 'Joint Correlation Violation Hazard'
              : visualizerMode === 'mixup_embedding'
                ? 'Naive Input-Space Categorical Validity'
                : 'Unscaled Raw Noise Distortion'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'feature_swapping'
              ? 'Downstream Regularization Benefit'
              : visualizerMode === 'mixup_embedding'
                ? 'Post-Embedding Semantic Manifold Validity'
                : 'Robust IQR-Scaled Perturbation'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Samples" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

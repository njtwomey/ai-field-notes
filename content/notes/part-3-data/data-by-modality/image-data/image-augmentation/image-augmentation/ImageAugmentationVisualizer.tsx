import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ImageAugmentationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'mixup_cutmix' | 'randaugment' | 'tta_tradeoff'>(
      [
        {
          value: 'mixup_cutmix',
          label: 'Sample Mixing: Mixup vs CutMix Manifold Interpolation (Zhang 2018, Yun 2019)',
        },
        { value: 'randaugment', label: 'Learned Policies: RandAugment Magnitude M vs Regularization (Cubuk 2020)' },
        { value: 'tta_tradeoff', label: 'Test-Time Augmentation (TTA): Ensembled Views K vs Accuracy Gain' },
      ],
      'mixup_cutmix',
      { label: 'augmentation strategy' },
    ),
    mixingBetaAlpha: slider(0.1, 1.5, 0.4, {
      step: 0.1,
      label: 'Beta distribution parameter alpha',
    }),
    randAugmentMag: slider(2, 30, 14, {
      step: 2,
      label: 'RandAugment distortion magnitude M (0 to 30)',
    }),
  })

  const { visualizerMode, mixingBetaAlpha, randAugmentMag } = state

  const results = useMemo(() => {
    if (visualizerMode === 'mixup_cutmix') {
      // Mixing coefficient lambda from 0.0 to 1.0 (11 steps)
      const lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      // Label target share for Class A: lambda * 100
      const labelShareA = lambdas.map((l) => l * 100)

      const currAlpha = mixingBetaAlpha

      // Theoretical average mixing probability density peak at boundaries vs center
      const isUshaped = currAlpha < 1.0

      return {
        x: lambdas.map((l) => Math.round(l * 100)),
        y: labelShareA,
        baseline: lambdas.map((l) => (1 - l) * 100), // Class B share
        stat1: `Beta parameter alpha = ${currAlpha.toFixed(1)}`,
        stat2: isUshaped ? 'U-shaped density (bimodal mix)' : 'Bell-shaped density (heavy blend)',
        stat3: 'Linear Label Interpolation',
        warning:
          currAlpha <= 0.4
            ? 'Optimal Mixup/CutMix: Samples mostly retain a dominant class with partial secondary context, avoiding uninformative 50/50 gray mixtures'
            : 'Heavy blending: High alpha values generate excessive ambiguous 50/50 hybrid images that slow down optimization',
        axisXLabel: 'Mixing Ratio Lambda (% Contribution of Image A)',
        axisYLabel: 'Target Label Probability Weight (%)',
        xRange: [-2, 102] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'randaugment') {
      // Magnitude M from 2 to 30
      const mags = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30]
      // Regularization benefit (validation accuracy gain %): peaks around M=12-16
      const accuracyGain = mags.map((m) => {
        const peak = 14
        const dist = Math.abs(m - peak)
        return Math.max(0, 2.8 - dist * 0.18)
      })
      // Semantic label distortion risk (% probability of corrupting digit/class semantics)
      const labelCorruptionRisk = mags.map((m) => Math.min(85, Math.pow(m / 30, 2.5) * 85))

      const currGain = Math.max(0, 2.8 - Math.abs(randAugmentMag - 14) * 0.18)
      const currCorruption = Math.min(85, Math.pow(randAugmentMag / 30, 2.5) * 85)

      return {
        x: mags,
        y: accuracyGain.map((g) => g * 10), // scaled for plot
        baseline: labelCorruptionRisk,
        stat1: `Magnitude M = ${randAugmentMag} / 30`,
        stat2: `Expected gain: +${currGain.toFixed(2)}% Top-1`,
        stat3: `Semantic drift risk: ${currCorruption.toFixed(1)}%`,
        warning:
          randAugmentMag > 22
            ? 'Excessive distortion magnitude: Extreme rotations and solarization destroy semantic class features, degrading accuracy'
            : randAugmentMag < 6
              ? 'Under-augmentation: Minimal distortion provides negligible regularization benefits'
              : 'Optimal RandAugment operating zone: Strong geometric and photometric diversity without semantic corruption',
        axisXLabel: 'RandAugment Transformation Magnitude M',
        axisYLabel: 'Metric Score (10x Accuracy Gain % / Corruption Risk %)',
        xRange: [1, 31] as [number, number],
        yRange: [0, 90] as [number, number],
      }
    } else {
      // Test-Time Augmentation views K: 1, 2, 4, 8, 12, 16
      const views = [1, 2, 4, 6, 8, 10, 12, 16]
      // Top-1 accuracy gain (%): logarithmic improvement
      const ttaGain = views.map((k) => (k === 1 ? 0 : Math.log2(k) * 0.45))
      // Latency multiplier: exactly K * 100%
      const latencyMult = views.map((k) => k)

      return {
        x: views,
        y: ttaGain.map((g) => g * 10), // scaled
        baseline: latencyMult.map((l) => l * 5), // scaled
        stat1: 'TTA Views: 1 to 16 views',
        stat2: 'At K=4: +0.9% accuracy gain',
        stat3: '4x Inference Latency Cost',
        warning:
          'TTA ensembles augmented views (flips, crops) to boost test accuracy, but multiplies inference compute cost by K times',
        axisXLabel: 'Ensembled Test-Time Augmentation Views (K)',
        axisYLabel: 'Index (10x Acc Gain % / 5x Latency Multiplier)',
        xRange: [0.5, 16.5] as [number, number],
        yRange: [0, 85] as [number, number],
      }
    }
  }, [visualizerMode, mixingBetaAlpha, randAugmentMag])

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
      title="Image Augmentation: Mixup/CutMix Interpolation, RandAugment Severity, and TTA Trade-offs"
      state={state}
      hoverReadout={false}
      caption="Data augmentation dynamics in computer vision: Linear label interpolation under Mixup and CutMix, finding the optimal distortion magnitude in RandAugment, and the latency trade-off of Test-Time Augmentation."
      readouts={
        <>
          <Readout label="Augmentation Setup" value={results.stat1} />
          <Readout label="Regularization Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'mixup_cutmix'
                ? 'Label Rule'
                : visualizerMode === 'randaugment'
                  ? 'Distortion Risk'
                  : 'Inference Cost'
            }
            value={results.stat3}
            color={visualizerMode === 'randaugment' ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
          <Readout label="Training Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Augmentation Regularization Profile">
        <Curve
          name={
            visualizerMode === 'mixup_cutmix'
              ? 'Class B Complementary Label Share (1 - lambda)'
              : visualizerMode === 'randaugment'
                ? 'Semantic Label Corruption Risk (%)'
                : 'Inference Latency Multiplier Index (5x)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'mixup_cutmix'
              ? 'Class A Primary Label Weight (lambda)'
              : visualizerMode === 'randaugment'
                ? 'Accuracy Gain Index (10x Top-1 %)'
                : 'TTA Accuracy Gain Index (10x Top-1 %)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Settings" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SelfSupervisedDataDesignVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'augmentation_pairs' | 'masking_ratio' | 'temperature_infonce'>(
      [
        { value: 'augmentation_pairs', label: 'Augmentation Composition & Linear Probe Yield (Chen 2020)' },
        { value: 'masking_ratio', label: 'Masked Autoencoder (MAE) Masking Ratio Trade-Off (He 2022)' },
        { value: 'temperature_infonce', label: 'InfoNCE Temperature tau & Gradient Penalty Hardness' },
      ],
      'augmentation_pairs',
      { label: 'design perspective' },
    ),
    maskingRatio: slider(15, 90, 75, {
      step: 5,
      label: 'masked autoencoder patch masking ratio (%)',
    }),
    temperatureTau: slider(0.05, 0.5, 0.1, {
      step: 0.05,
      label: 'contrastive InfoNCE temperature tau',
    }),
  })

  const { visualizerMode, maskingRatio, temperatureTau } = state

  const results = useMemo(() => {
    if (visualizerMode === 'augmentation_pairs') {
      // 5 composition configurations
      // 1: Random Crop Only
      // 2: Color Jitter Only
      // 3: Crop + Color Jitter (The critical synergy)
      // 4: Crop + Color + Gaussian Blur
      // 5: Full Pipeline (Crop + Color + Blur + Solarize)
      const configs = [1, 2, 3, 4, 5]
      const probeAcc = [32.4, 14.2, 56.3, 64.8, 69.3]
      const colorShortcutRisk = [88.0, 12.0, 9.5, 4.0, 1.5]

      return {
        x: configs,
        y: probeAcc,
        baseline: colorShortcutRisk,
        stat1: 'Crop + Color synergy: +23.9% jump',
        stat2: 'Full composition yield: 69.3% linear probe',
        stat3: 'Shortcut elimination: <2% color cheat',
        warning: 'Crop alone allows trivial matching of global color histograms; color jitter is mandatory',
        axisXLabel: 'Augmentation Pair Strategy (1: Crop, 2: Jitter, 3: Crop+Jitter, 4: +Blur, 5: Full)',
        axisYLabel: 'Linear Probe Accuracy vs Shortcut Risk (%)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, 95] as [number, number],
      }
    } else if (visualizerMode === 'masking_ratio') {
      // Masking ratio from 15% to 90%
      const ratios = [15, 30, 45, 60, 70, 75, 80, 85, 90]
      // Fine-tuning downstream accuracy: peaks at 75% for MAE vision models
      const ftAccuracy = ratios.map((r) => {
        const dist = r - 75
        return 83.5 - 0.015 * Math.pow(dist, 2)
      })
      // Computational speedup (encoder processes only 1 - r fraction of tokens)
      const speedup = ratios.map((r) => (100 / (100 - r)) * 15)

      const currAcc = 83.5 - 0.015 * Math.pow(maskingRatio - 75, 2)
      const currSpeed = (100 / (100 - maskingRatio)).toFixed(1)

      return {
        x: ratios,
        y: ftAccuracy,
        baseline: speedup,
        stat1: `Masking ratio: ${maskingRatio}%`,
        stat2: `Downstream accuracy: ${currAcc.toFixed(1)}%`,
        stat3: `Encoder speedup: ${currSpeed}x FLOPs reduction`,
        warning:
          maskingRatio < 40
            ? 'Low masking ratio: trivial local pixel interpolation stalls high-level abstraction'
            : maskingRatio > 85
              ? 'Excessive masking: insufficient context causes semantic reconstruction collapse'
              : 'Optimal 75% vision ratio: forces holistic semantic scene understanding',
        axisXLabel: 'Pretext Masking Ratio (%)',
        axisYLabel: 'Fine-Tuning Accuracy & Encoder Speedup Factor',
        xRange: [10, 95] as [number, number],
        yRange: [0, 92] as [number, number],
      }
    } else {
      // Cosine similarity with negative instances from -0.5 to 0.95
      const sims = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
      const tau = temperatureTau

      // InfoNCE gradient weight: exp(sim / tau) normalized
      const weights = sims.map((s) => {
        const expVal = Math.exp(s / tau)
        return Math.min(100, expVal * 0.01)
      })
      const uniformFlat = sims.map(() => 10.0)

      return {
        x: sims,
        y: weights,
        baseline: uniformFlat,
        stat1: `Temperature tau = ${tau.toFixed(2)}`,
        stat2: `Hardest negative weight: ${weights[9].toFixed(1)}x`,
        stat3: 'Negative hardness gradient scaling',
        warning:
          tau < 0.08
            ? 'Extreme temperature: over-concentrates on hardest negatives, vulnerable to false negatives'
            : tau > 0.3
              ? 'High temperature: gradient weights dilute across easy negatives, weakening representation margins'
              : 'Balanced temperature: enforces uniform hyperspherical distribution without collapse',
        axisXLabel: 'Negative Embedding Cosine Similarity with Anchor',
        axisYLabel: 'Relative InfoNCE Gradient Penalty Weight',
        xRange: [-0.5, 1.0] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, maskingRatio, temperatureTau])

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
      title="Self-Supervised Data Design & Invariance Inspector"
      state={state}
      hoverReadout={false}
      caption="Evaluating self-supervised data mechanics: The non-linear synergy of augmentation pair compositions, optimal masking ratios in masked autoencoding, and InfoNCE temperature dynamics."
      readouts={
        <>
          <Readout label="Pretext Parameter" value={results.stat1} />
          <Readout label="Representation Metric" value={results.stat2} />
          <Readout label="Optimization Dynamic" value={results.stat3} />
          <Readout
            label="Design Directive"
            value={results.warning}
            color={
              (visualizerMode === 'masking_ratio' && (maskingRatio < 40 || maskingRatio > 85)) ||
              (visualizerMode === 'temperature_infonce' && temperatureTau < 0.08)
                ? 'var(--red-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Self-Supervised Representation Frontier">
        <Curve
          name={
            visualizerMode === 'augmentation_pairs'
              ? 'Color Histogram Shortcut Risk'
              : visualizerMode === 'masking_ratio'
                ? 'Encoder Computational Acceleration Factor'
                : 'Uniform Gradient Weight Baseline'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'augmentation_pairs'
              ? 'Linear Probe Classification Accuracy'
              : visualizerMode === 'masking_ratio'
                ? 'Downstream Fine-Tuning Accuracy'
                : 'Negative Penalty Gradient Density'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Knots" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

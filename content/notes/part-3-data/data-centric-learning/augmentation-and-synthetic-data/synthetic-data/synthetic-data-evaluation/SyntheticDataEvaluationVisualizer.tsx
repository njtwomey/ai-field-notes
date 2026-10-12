import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SyntheticDataEvaluationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'pr_frontier' | 'tstr_matrix' | 'memorization_nndr'>(
      [
        { value: 'pr_frontier', label: 'Precision (Fidelity) vs Recall (Diversity) Frontier (Sajjadi 2018)' },
        { value: 'tstr_matrix', label: 'TSTR vs TRTR Downstream Utility Matrix' },
        { value: 'memorization_nndr', label: 'Nearest Neighbor Distance Ratio (NNDR) & Memorisation Risk' },
      ],
      'pr_frontier',
      { label: 'evaluation perspective' },
    ),
    truncationPsi: slider(0.2, 1.0, 0.7, {
      step: 0.1,
      label: 'generator truncation / temperature psi',
    }),
    nndrThreshold: slider(0.1, 0.6, 0.25, {
      step: 0.05,
      label: 'memorisation distance cutoff threshold',
    }),
  })

  const { visualizerMode, truncationPsi, nndrThreshold } = state

  const results = useMemo(() => {
    if (visualizerMode === 'pr_frontier') {
      // Recall values from 20% to 95%
      const recalls = [20, 30, 40, 50, 60, 70, 80, 85, 90, 95]
      // SOTA Diffusion/Autoregressive frontier (convex trade-off)
      // High precision at low recall (truncated psi); lower precision at high recall
      const sotaPrecision = recalls.map((r) => Math.max(30, 96 - 0.007 * Math.pow(r - 20, 2)))
      // Mode-dropping GAN baseline: precision high initially, collapses past 55% recall
      const ganPrecision = recalls.map((r) => {
        if (r <= 50) return 92 - (r - 20) * 0.4
        return Math.max(10, 80 - Math.pow(r - 50, 1.8) * 1.6)
      })

      // Point at selected truncation psi
      // Low psi (0.2): Precision ~95%, Recall ~35% (Mode dropping)
      // High psi (1.0): Precision ~65%, Recall ~92% (High diversity, lower fidelity)
      const currRecall = 30 + (truncationPsi - 0.2) * 75
      const currPrecision = Math.max(30, 96 - 0.007 * Math.pow(currRecall - 20, 2))

      return {
        x: recalls,
        y: sotaPrecision,
        baseline: ganPrecision,
        stat1: `Truncation psi = ${truncationPsi.toFixed(1)}`,
        stat2: `Generative Precision (Fidelity): ${currPrecision.toFixed(1)}%`,
        stat3: `Generative Recall (Diversity): ${currRecall.toFixed(1)}%`,
        warning:
          truncationPsi <= 0.4
            ? 'Severe mode dropping: high image fidelity but omits major distribution modes'
            : truncationPsi >= 0.9
              ? 'High diversity, but tail generations contain unrealistic artifacts'
              : 'Balanced operational regime: high fidelity with robust support coverage',
        axisXLabel: 'Generative Recall (Diversity & Mode Coverage) [%]',
        axisYLabel: 'Generative Precision (Fidelity & Realism) [%]',
        xRange: [15, 100] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'tstr_matrix') {
      // 4 utility regimes:
      // 1: TRTR (Train on Real, Test on Real - Gold standard)
      // 2: TSTR (Train on Synthetic, Test on Real - True utility)
      // 3: TRTS (Train on Real, Test on Synthetic - Coverage check)
      // 4: TSTS (Train on Synthetic, Test on Synthetic - Dangerous illusion)
      const regimes = [1, 2, 3, 4]
      const accuracies = [88.5, 82.2, 79.4, 94.8]
      const trtrCeiling = regimes.map(() => 88.5)

      return {
        x: regimes,
        y: accuracies,
        baseline: trtrCeiling,
        stat1: 'TRTR Baseline: 88.5% accuracy',
        stat2: 'TSTR Utility: 82.2% (92.9% retention)',
        stat3: 'TSTS Overconfidence: 94.8% (+6.3% gap)',
        warning: 'CRITICAL TRAP: TSTS yields artificially high scores; always evaluate TSTR on pristine real data',
        axisXLabel: 'Evaluation Paradigm (1: TRTR, 2: TSTR, 3: TRTS, 4: TSTS)',
        axisYLabel: 'Downstream Task Evaluation Accuracy (%)',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [65, 100] as [number, number],
      }
    } else {
      // NNDR distance ratio bins from 0.05 to 0.95
      const nndrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      // Clean generator distribution: unimodal around 0.55
      const cleanDensity = nndrs.map(
        (d) => (1 / (0.16 * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((d - 0.55) / 0.16, 2)) * 30,
      )
      // Overfitted generator copying data: sharp spike at d < 0.15
      const overfittedDensity = nndrs.map((d) => {
        if (d <= 0.2) return 55 * Math.exp(-d / 0.06)
        return 12 * Math.exp(-d / 0.3)
      })

      return {
        x: nndrs.map((d) => d * 100),
        y: cleanDensity,
        baseline: overfittedDensity,
        stat1: 'Clean generator mode: NNDR ~ 0.55',
        stat2: 'Overfitted copy peak: NNDR < 0.15',
        stat3: `Cutoff threshold: ${(nndrThreshold * 100).toFixed(0)}%`,
        warning: 'Samples with NNDR < 0.2 represent verbatim copying of training subjects, violating privacy',
        axisXLabel: 'Nearest Neighbor Distance Ratio (NNDR) [%]',
        axisYLabel: 'Empirical Sample Frequency Density',
        xRange: [0, 100] as [number, number],
        yRange: [0, 85] as [number, number],
      }
    }
  }, [visualizerMode, truncationPsi, nndrThreshold])

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
      title="Synthetic Data Evaluation & Fidelity-Diversity Inspector"
      state={state}
      hoverReadout={false}
      caption="Quantifying synthetic data quality: The Precision (fidelity) versus Recall (diversity) frontier, Train on Synthetic Test on Real (TSTR) utility comparisons, and NNDR memorisation detection."
      readouts={
        <>
          <Readout label="Evaluation Domain" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Comparative Leverage" value={results.stat3} />
          <Readout
            label="Evaluation Notice"
            value={results.warning}
            color={
              (visualizerMode === 'pr_frontier' && truncationPsi <= 0.4) || visualizerMode === 'tstr_matrix'
                ? 'var(--blue-600)'
                : 'var(--emerald-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Evaluation Trajectory">
        <Curve
          name={
            visualizerMode === 'pr_frontier'
              ? 'Mode-Dropping GAN Baseline'
              : visualizerMode === 'tstr_matrix'
                ? 'Real-Data Performance Ceiling (TRTR)'
                : 'Overfitted Memorisation Profile'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'pr_frontier'
              ? 'SOTA Generative Precision-Recall Frontier'
              : visualizerMode === 'tstr_matrix'
                ? 'Empirical Task Accuracy'
                : 'Generalizing Synthetic Generator'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Knots" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

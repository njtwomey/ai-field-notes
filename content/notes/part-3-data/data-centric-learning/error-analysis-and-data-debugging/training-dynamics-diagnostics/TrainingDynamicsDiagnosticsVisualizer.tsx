import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TrainingDynamicsDiagnosticsVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'cartography_map' | 'epoch_trajectories' | 'forgetting_distribution'>(
      [
        { value: 'cartography_map', label: 'Dataset Cartography Plane (Swayamdipta 2020)' },
        { value: 'epoch_trajectories', label: 'Per-Example True-Class Probabilities Across Epochs' },
        { value: 'forgetting_distribution', label: 'Forgetting Events & Unforgettable Data (Toneva 2019)' },
      ],
      'cartography_map',
      { label: 'diagnostic view' },
    ),
    selectedCohort: choice<'easy' | 'ambiguous' | 'hard' | 'all'>(
      [
        { value: 'all', label: 'Full Dataset Landscape' },
        { value: 'ambiguous', label: 'Ambiguous Slice (High Generalization Value)' },
        { value: 'hard', label: 'Hard-to-Learn Slice (Enriched for Label Noise)' },
        { value: 'easy', label: 'Easy-to-Learn Slice (Prunable Canonical Data)' },
      ],
      'all',
      { label: 'cartography cohort' },
    ),
    epochCount: slider(5, 40, 20, {
      step: 5,
      label: 'training observation epochs E',
    }),
  })

  const { visualizerMode, selectedCohort, epochCount } = state

  const results = useMemo(() => {
    if (visualizerMode === 'cartography_map') {
      // Confidence mu from 0.05 to 0.95
      const confidences = [0.05, 0.12, 0.22, 0.35, 0.45, 0.52, 0.62, 0.72, 0.82, 0.9, 0.96]
      // In cartography, variability peaks at intermediate confidence (inverted parabola)
      const maxVar = 0.42 * Math.sqrt(Math.min(1.0, epochCount / 20))
      const variabilities = confidences.map((c) => 4 * maxVar * c * (1 - c))

      // Envelope baseline for easy/hard division
      const envelope = confidences.map(() => 0.15)

      let stat1 = 'Cartography coordinate: (mu_i, sigma_i)'
      let stat2 = 'Total instances: 100% indexed'
      let stat3 = 'Ambiguous band: sigma > 0.25'
      let warning = 'Track example trajectories over training to expose data quality taxonomy'

      if (selectedCohort === 'ambiguous') {
        stat1 = 'Ambiguous slice: ~32% of dataset'
        stat2 = 'OOD transfer: drives generalization'
        stat3 = 'Boundary uncertainty: sigma_i in [0.25, 0.45]'
        warning = 'Ambiguous examples contain the highest gradient utility for out-of-domain transfer'
      } else if (selectedCohort === 'hard') {
        stat1 = 'Hard slice: ~14% of dataset'
        stat2 = 'Noise enrichment: >60% mislabeled'
        stat3 = 'Low confidence: mu_i < 0.25'
        warning = 'Hard-to-learn examples are prime candidates for label verification and cleaning'
      } else if (selectedCohort === 'easy') {
        stat1 = 'Easy slice: ~54% of dataset'
        stat2 = 'Pruning tolerance: up to 50% prunable'
        stat3 = 'Canonical stability: mu_i > 0.80'
        warning = 'Easy examples are learned immediately and contribute near-zero late training gradient'
      }

      return {
        x: confidences.map((c) => c * 100),
        y: variabilities.map((v) => v * 100),
        baseline: envelope.map((e) => e * 100),
        stat1,
        stat2,
        stat3,
        warning,
        axisXLabel: 'Model Confidence mu_i: Mean True-Class Probability (%)',
        axisYLabel: 'Variability sigma_i: Std Dev of Probability (%)',
        xRange: [0, 100] as [number, number],
        yRange: [0, 50] as [number, number],
      }
    } else if (visualizerMode === 'epoch_trajectories') {
      // Epochs from 1 to 20
      const epochs = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
      // 1. Easy example: rapidly hits 98%
      const easyP = epochs.map((e) => 100 * (1 - Math.exp(-e / 1.2)))
      // 2. Ambiguous example: oscillates with high frequency
      const ambigP = epochs.map((e) => 50 + 28 * Math.sin(e * 1.1) * Math.exp(-e / 30))
      // 3. Mislabeled example: stays low, late memorization bump
      const mislabeledP = epochs.map((e) => 8 + (e > 14 ? (35 * (e - 14)) / 6 : 4 * Math.sin(e)))

      let currentCurve = ambigP
      let baselineCurve = easyP
      if (selectedCohort === 'hard') {
        currentCurve = mislabeledP
        baselineCurve = easyP
      } else if (selectedCohort === 'easy') {
        currentCurve = easyP
        baselineCurve = mislabeledP
      }

      return {
        x: epochs,
        y: currentCurve,
        baseline: baselineCurve,
        stat1: 'Canonical: p > 0.90 within 3 epochs',
        stat2: 'Ambiguous: oscillates across epochs',
        stat3: 'Label error: resists fit until memorization',
        warning: 'Sudden late memorization of low-probability examples signals overfitting on label noise',
        axisXLabel: 'Training Epoch e',
        axisYLabel: 'True-Class Softmax Probability P(y = y_i) [%]',
        xRange: [1, 21] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Forgetting events 0 to 10
      const events = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Proportions in dataset: Unforgettable (0) is ~42%, then rapid decay
      const proportions = [42.0, 18.0, 12.0, 8.5, 6.0, 4.5, 3.2, 2.3, 1.6, 1.1, 0.8]
      const pruneThresholdLine = events.map((ev) => (ev === 0 ? 42.0 : 0))

      return {
        x: events,
        y: proportions,
        baseline: pruneThresholdLine,
        stat1: 'Unforgettable (F = 0): 42.0% of data',
        stat2: 'Frequently forgotten (F >= 4): 19.5%',
        stat3: 'Pruning 30% unforgettable: zero accuracy drop',
        warning: 'Unforgettable examples can be safely pruned to drastically cut epoch training wall-time',
        axisXLabel: 'Count of Forgetting Events During Training',
        axisYLabel: 'Dataset Proportion (%)',
        xRange: [-0.5, 10.5] as [number, number],
        yRange: [0, 48] as [number, number],
      }
    }
  }, [visualizerMode, selectedCohort, epochCount])

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
      title="Training Dynamics & Dataset Cartography Diagnostics"
      state={state}
      hoverReadout={false}
      caption="Evaluating per-example learning dynamics: Dataset cartography coordinate partitions (easy, ambiguous, hard-to-learn), epoch-by-epoch true class probabilities, and forgetting event distributions."
      readouts={
        <>
          <Readout label="Dynamics Coordinate" value={results.stat1} />
          <Readout label="Generalization / Utility" value={results.stat2} />
          <Readout label="Diagnostic Threshold" value={results.stat3} />
          <Readout
            label="Optimization Insight"
            value={results.warning}
            color={
              selectedCohort === 'hard'
                ? 'var(--red-600)'
                : selectedCohort === 'ambiguous'
                  ? 'var(--emerald-600)'
                  : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Training Trajectory Space">
        <Curve
          name={
            visualizerMode === 'cartography_map'
              ? 'Low Variability Partition Threshold'
              : visualizerMode === 'epoch_trajectories'
                ? 'Reference Trajectory'
                : 'Unforgettable Cutoff Mark'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'cartography_map'
              ? 'Variability Envelope sigma(mu)'
              : visualizerMode === 'epoch_trajectories'
                ? 'Active Example Probability Path'
                : 'Forgetting Event Distribution'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Indexed Instances" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function BaselineProbesVisualizer() {
  const state = useFigureState({
    probeType: choice<'single_feature' | 'shuffled_labels'>(
      [
        { value: 'single_feature', label: 'Single-Feature Probing (Detecting Leaky Super-Features)' },
        { value: 'shuffled_labels', label: 'Shuffled-Label Permutation Probe (Testing Null Hygiene)' },
      ],
      'single_feature',
      { label: 'diagnostic probe' },
    ),
    leakageStrength: slider(0.0, 1.0, 0.6, {
      step: 0.1,
      label: 'target leakage injection severity into feature 4',
    }),
  })

  const results = useMemo(() => {
    // 6 candidate features:
    // F1: Age (58% acc)
    // F2: Prior visits (64% acc)
    // F3: Device type (52% acc)
    // F4: "Leaky Column" (e.g., account_closed_reason / post-event timestamp)
    // F5: Zip code median income (61% acc)
    // F6: Random Gaussian noise (50% acc)
    const features = [1, 2, 3, 4, 5, 6]
    const leak = state.leakageStrength

    const singleFeatureAcc = [58.0, 64.0, 52.0, 50.0 + 49.5 * leak, 61.0, 50.2]

    // Shuffled label accuracy on training vs validation
    // If leak > 0 (e.g., cross-split duplicate row IDs), validation accuracy on shuffled labels > 50%
    const shuffledValAcc = 50.0 + 38.0 * leak
    const shuffledTrainAcc = 50.0 + 48.0 * leak

    const f4Acc = singleFeatureAcc[3]
    const isLeaky = f4Acc > 88.0

    // Threshold suspicion line at 85%
    const suspicionLine = features.map(() => 85.0)

    const plotY =
      state.probeType === 'single_feature' ? singleFeatureAcc : [50.0, 50.0, 50.0, shuffledValAcc, 50.0, 50.0]

    return {
      features,
      singleFeatureAcc,
      shuffledValAcc,
      shuffledTrainAcc,
      f4Acc,
      isLeaky,
      suspicionLine,
      plotY,
    }
  }, [state.leakageStrength, state.probeType])

  const axisX = useAxis({
    label: 'Feature Index (1: Age, 2: Visits, 3: Device, 4: Leaky Candidate, 5: Income, 6: Noise)',
    range: [0.5, 6.5],
  })

  const axisY = useAxis({
    label: 'Validation Accuracy (%)',
    range: [45, 102],
  })

  return (
    <Figure
      title="Baseline Models as Diagnostic Probes: Detecting Leakage and Trivialities"
      state={state}
      hoverReadout={false}
      caption="A simple 1-variable baseline acts as a sensor for data integrity. If Feature 4 achieves 95%+ accuracy in isolation, it is almost certainly a post-event label leak rather than a true predictive feature. Shuffling labels confirms whether cross-split entity leakage exists."
      readouts={
        <>
          <Readout label="Feature 4 Probe Accuracy" value={`${results.f4Acc.toFixed(1)}%`} />
          <Readout
            label="Leakage Alarm"
            value={results.isLeaky ? 'CRITICAL LEAK DETECTED (Feature 4)' : 'Normal Signal Range'}
          />
          <Readout
            label="Shuffled-Label Probe"
            value={`${results.shuffledValAcc.toFixed(1)}% val accuracy (${results.shuffledValAcc > 55 ? 'Memory Leak!' : 'Clean Null'})`}
          />
          <Readout label="Noise Baseline (F6)" value="50.2% (Matches random chance)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Diagnostic Probe Accuracy Across Features">
        <Points name="Probe Accuracy" x={results.features} y={results.plotY} slot={results.isLeaky ? 1 : 0} size={8} />
        <Curve
          name="Suspicion Threshold (85%)"
          x={results.features}
          y={results.suspicionLine}
          slot={2}
          dashed
          width={2}
        />
      </Plot>
    </Figure>
  )
}

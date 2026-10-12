import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ExplorationHygieneVisualizer() {
  const state = useFigureState({
    hygieneProtocol: choice<'unhygienic_full_eda' | 'hygienic_vaulted'>(
      [
        { value: 'unhygienic_full_eda', label: 'Unhygienic EDA (Explored full data before split: Snooping leakage)' },
        { value: 'hygienic_vaulted', label: 'Hygienic Protocol (Test set vaulted in cold storage before EDA)' },
      ],
      'unhygienic_full_eda',
      { label: 'protocol discipline' },
    ),
    forkingPaths: slider(1, 20, 8, {
      step: 1,
      label: 'exploratory decisions / feature transformations tested (K)',
    }),
  })

  const results = useMemo(() => {
    const kChoices = state.forkingPaths
    const alpha = 0.05 // nominal per-comparison significance level

    // Cumulative false positive discovery probability under null: 1 - (1 - alpha)^K
    const pFalseDiscovery = 1.0 - Math.pow(1.0 - alpha, kChoices)

    // Sweep K across 1..20
    const kGrid: number[] = []
    const unhygienicCurve: number[] = []
    const hygienicCurve: number[] = []

    for (let k = 1; k <= 20; k += 1) {
      kGrid.push(k)
      // Unhygienic snooping probability grows exponentially
      unhygienicCurve.push((1.0 - Math.pow(1.0 - alpha, k)) * 100.0)
      // Hygienic maintains strict nominal control because test set is unobserved
      hygienicCurve.push(alpha * 100.0)
    }

    const currentY = state.hygieneProtocol === 'unhygienic_full_eda' ? pFalseDiscovery * 100.0 : alpha * 100.0

    const reportedAccuracy = state.hygieneProtocol === 'unhygienic_full_eda' ? 50.0 + 35.0 * pFalseDiscovery : 50.0 // under true null

    const trueAccuracy = 50.0 // random chance under null

    return {
      kChoices,
      pFalseDiscovery: pFalseDiscovery * 100.0,
      kGrid,
      unhygienicCurve,
      hygienicCurve,
      currentY,
      reportedAccuracy,
      trueAccuracy,
      optimismInflation: reportedAccuracy - trueAccuracy,
    }
  }, [state.forkingPaths, state.hygieneProtocol])

  const axisX = useAxis({
    label: 'Number of Exploratory Forking Decisions Tested (K)',
    range: [0.8, 20.2],
  })

  const axisY = useAxis({
    label: 'False Discovery Probability (%)',
    range: [0, 75],
  })

  return (
    <Figure
      title="Exploration Hygiene: The Garden of Forking Paths and Data Snooping"
      state={state}
      hoverReadout={false}
      caption="When an analyst inspects the full dataset before splitting, decisions regarding outlier clipping, binning, and feature selection are unconsciously optimized to fit test rows. As the number of exploratory forks K increases, finding spurious patterns becomes almost certain."
      readouts={
        <>
          <Readout label="Exploratory Paths (K)" value={`${results.kChoices} transformations`} />
          <Readout label="Spurious Pattern Risk" value={`${results.currentY.toFixed(1)}% probability`} />
          <Readout
            label="Reported vs True Test Accuracy"
            value={`${results.reportedAccuracy.toFixed(1)}% vs ${results.trueAccuracy.toFixed(1)}%`}
          />
          <Readout label="Optimism Leakage Gap" value={`+${results.optimismInflation.toFixed(1)}% illusory gain`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Exploratory Flexibility vs Spurious Discovery Rate">
        <Curve
          name="Unhygienic Snooping Discovery Risk"
          x={results.kGrid}
          y={results.unhygienicCurve}
          slot={2}
          width={2.5}
        />
        <Curve
          name="Hygienic Cold-Storage Benchmark (Alpha = 5%)"
          x={results.kGrid}
          y={results.hygienicCurve}
          slot={0}
          dashed
          width={2}
        />
        <Points
          name="Current Protocol Operating Point"
          x={[results.kChoices]}
          y={[results.currentY]}
          slot={state.hygieneProtocol === 'unhygienic_full_eda' ? 2 : 0}
          size={8}
        />
      </Plot>
    </Figure>
  )
}

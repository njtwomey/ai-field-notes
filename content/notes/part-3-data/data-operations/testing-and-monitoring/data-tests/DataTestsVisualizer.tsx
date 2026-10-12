import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataTestsVisualizer() {
  const state = useFigureState({
    assertionType: choice<'split_overlap' | 'join_invariant' | 'golden_regression'>(
      [
        { value: 'split_overlap', label: 'Split-Overlap Invariant (Assert Train ∩ Test == ∅: Breck et al. 2017)' },
        { value: 'join_invariant', label: 'Join Multiplicity Invariant (Assert RowCount(Out) == RowCount(Base))' },
        { value: 'golden_regression', label: 'Golden Sample Regression (Assert |f(x) - f_golden(x)| < 1e-6)' },
      ],
      'split_overlap',
      { label: 'data pipeline unit test' },
    ),
    corruptionLevel: slider(0, 10, 2, {
      step: 1,
      label: 'synthetic pipeline defect injection rate (%)',
    }),
  })

  const results = useMemo(() => {
    // Generate detection probability curves across defect injection p in [0, 10]%
    const pGrid: number[] = []
    const detectionProbGrid: number[] = []

    for (let p = 0; p <= 10; p += 1) {
      pGrid.push(p)
      let prob = 0
      if (p === 0) {
        prob = 0.0 // Zero defects, test passes cleanly
      } else {
        // Statistical detection power scales sharply with sample size N
        prob = Math.min(100, 100.0 * (1.0 - Math.pow(1.0 - p / 100, 150)))
      }
      detectionProbGrid.push(prob)
    }

    const currentP = state.corruptionLevel
    let currentDetectionProb = 0
    if (currentP === 0) {
      currentDetectionProb = 0.0
    } else {
      currentDetectionProb = Math.min(100, 100.0 * (1.0 - Math.pow(1.0 - currentP / 100, 150)))
    }

    let ciStatus = ''
    let assertionMessage = ''

    if (currentP === 0) {
      ciStatus = 'CI Green (All assertions passed)'
      assertionMessage = 'Zero violations detected'
    } else {
      ciStatus = 'CI Blocked (Pipeline aborted prior to training)'
      if (state.assertionType === 'split_overlap') {
        assertionMessage = `Leakage caught: ${currentP}% contamination between train and test splits`
      } else if (state.assertionType === 'join_invariant') {
        assertionMessage = `Join explosion caught: row count increased by ${currentP * 1.5}%`
      } else {
        assertionMessage = `Regression detected: ${currentP} golden sample feature values drifted`
      }
    }

    return {
      pGrid,
      detectionProbGrid,
      currentP,
      currentDetectionProb,
      ciStatus,
      assertionMessage,
    }
  }, [state.assertionType, state.corruptionLevel])

  const axisX = useAxis({
    label: 'injected pipeline defect rate (%)',
    range: [0, 10],
  })
  const axisY = useAxis({
    label: 'pipeline test detection & catch rate (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Automated data pipeline unit testing and CI assertion dynamics"
      state={state}
      hoverReadout={false}
      caption="Evaluating unit test assertions for data engineering pipelines (Breck et al. 2017). Split-overlap assertions verify that training and test partitions share zero overlapping entity identifiers, preventing data leakage before model training commences. Join multiplicity tests verify that dimensional joins do not silently duplicate rows via 1-to-many Cartesian expansion. Golden sample assertions verify numerical determinism across software dependency updates."
      readouts={
        <>
          <Readout label="CI pipeline status" value={results.ciStatus} />
          <Readout label="test assertion result" value={results.assertionMessage} />
          <Readout
            label="defect interception power"
            value={`${results.currentDetectionProb.toFixed(1)}% caught in CI`}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Defect interception rate vs defect injection rate">
        <Curve
          name="defect catch probability (%)"
          x={results.pGrid}
          y={results.detectionProbGrid}
          slot={results.currentP > 0 ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current defect rate"
          x={[results.currentP]}
          y={[results.currentDetectionProb]}
          slot={results.currentP > 0 ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}

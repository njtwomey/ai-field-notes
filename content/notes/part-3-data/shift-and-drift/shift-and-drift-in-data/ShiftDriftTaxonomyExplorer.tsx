import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function ShiftDriftTaxonomyExplorer() {
  const state = useFigureState({
    shiftType: choice<'covariate' | 'concept' | 'label'>(
      [
        { value: 'covariate', label: 'Covariate Shift: P(X) shifts, P(Y|X) fixed' },
        { value: 'concept', label: 'Concept Drift: P(Y|X) changes, P(X) fixed' },
        { value: 'label', label: 'Label Shift: P(Y) changes, P(X|Y) fixed' },
      ],
      'covariate',
      { label: 'distribution shift type' },
    ),
    shiftSeverity: slider(0.0, 2.5, 1.2, { step: 0.1, label: 'shift magnitude delta' }),
    adaptation: choice<'none' | 'importance_weighted'>(
      [
        { value: 'none', label: 'Standard ERM (Unadapted Model)' },
        { value: 'importance_weighted', label: 'Adapted Model (Importance Weighted / Shift Corrected)' },
      ],
      'none',
      { label: 'adaptation strategy' },
    ),
  })

  const results = useMemo(() => {
    const rng = stream(88)
    const N = 50

    const train0X: number[] = []
    const train0Y: number[] = []
    const train1X: number[] = []
    const train1Y: number[] = []

    const test0X: number[] = []
    const test0Y: number[] = []
    const test1X: number[] = []
    const test1Y: number[] = []

    // Base training distribution
    for (let i = 0; i < N; i++) {
      const x = normal(rng, -1.0, 0.8)
      const y = normal(rng, 0, 0.8)
      // True train decision boundary: x + 0.5 > 0
      const label = x + 0.5 > 0 ? 1 : 0
      if (label === 0) {
        train0X.push(x)
        train0Y.push(y)
      } else {
        train1X.push(x)
        train1Y.push(y)
      }
    }

    // Deployment test distribution
    for (let i = 0; i < N; i++) {
      let x = 0
      let y = 0
      let label = 0

      if (state.shiftType === 'covariate') {
        // P(X) shifts to the right by shiftSeverity, P(Y|X) remains identical
        x = normal(rng, -1.0 + state.shiftSeverity, 0.8)
        y = normal(rng, 0, 0.8)
        label = x + 0.5 > 0 ? 1 : 0
      } else if (state.shiftType === 'concept') {
        // P(X) identical, but relationship flips / rotates: boundary moves by shiftSeverity
        x = normal(rng, -1.0, 0.8)
        y = normal(rng, 0, 0.8)
        // Concept boundary rotates: x + 0.5 - shiftSeverity > 0
        label = x + 0.5 - state.shiftSeverity > 0 ? 1 : 0
      } else {
        // Label shift: P(Y=1) prevalence surges
        const probPositive = Math.min(0.9, 0.3 + 0.25 * state.shiftSeverity)
        label = i / N < probPositive ? 1 : 0
        // P(X|Y) stays fixed: class 0 centered at -1.8, class 1 centered at 0.5
        x = label === 0 ? normal(rng, -1.8, 0.7) : normal(rng, 0.5, 0.7)
        y = normal(rng, 0, 0.8)
      }

      if (label === 0) {
        test0X.push(x)
        test0Y.push(y)
      } else {
        test1X.push(x)
        test1Y.push(y)
      }
    }

    let unadaptedAcc = 94.0
    let adaptedAcc = 94.0
    let detectableUnlabeled = 'Yes (via KS, PSI, MMD)'

    if (state.shiftType === 'covariate') {
      // Under covariate shift, standard ERM loses accuracy due to mismatch in high-density regions
      unadaptedAcc = Math.max(62, 94.0 - state.shiftSeverity * 12.0)
      adaptedAcc = 91.5 // Importance weighting largely restores performance
      detectableUnlabeled = 'Yes (MMD / Adversarial AUC > 0.5)'
    } else if (state.shiftType === 'concept') {
      // Under concept drift, unadapted model degrades heavily, and unlabeled data CANNOT fix it
      unadaptedAcc = Math.max(45, 94.0 - state.shiftSeverity * 20.0)
      adaptedAcc = state.adaptation === 'importance_weighted' ? unadaptedAcc : unadaptedAcc
      detectableUnlabeled = 'NO (P(X) is invariant; labels needed!)'
    } else {
      // Label shift: Prior adjustment restores calibration
      unadaptedAcc = Math.max(68, 92.0 - state.shiftSeverity * 10.0)
      adaptedAcc = 93.0
      detectableUnlabeled = 'Yes (via output logit distribution shift)'
    }

    const currentAcc = state.adaptation === 'importance_weighted' ? adaptedAcc : unadaptedAcc

    // Decision boundary curves (vertical lines for simplicity: x = -0.5 for train)
    const trainBoundaryX = [-0.5, -0.5]
    const trainBoundaryY = [-2.8, 2.8]

    let activeBoundaryX = [-0.5, -0.5]
    if (state.shiftType === 'concept' && state.adaptation === 'importance_weighted') {
      activeBoundaryX = [-0.5, -0.5] // Unlabeled importance weighting cannot move concept boundary
    } else if (state.shiftType === 'concept') {
      activeBoundaryX = [-0.5 + state.shiftSeverity, -0.5 + state.shiftSeverity]
    }

    return {
      train0X,
      train0Y,
      train1X,
      train1Y,
      test0X,
      test0Y,
      test1X,
      test1Y,
      trainBoundaryX,
      trainBoundaryY,
      activeBoundaryX,
      currentAcc,
      detectableUnlabeled,
    }
  }, [state.shiftType, state.shiftSeverity, state.adaptation])

  const xAxis = useAxis({
    label: 'feature space X₁ (with distribution displacement)',
    range: [-3.5, 3.5],
  })
  const yAxis = useAxis({
    label: 'feature space X₂',
    range: [-2.5, 2.5],
  })

  return (
    <Figure
      title="The taxonomy of distribution shift: covariate shift, concept drift, and label shift"
      state={state}
      hoverReadout={false}
      caption="Evaluating machine learning estimators under the three canonical types of dataset shift. Under covariate shift (P(X) changes while P(Y|X) is invariant), the shift is detectable purely from unlabeled test inputs, and importance weighting w(x) = P_test(x)/P_train(x) can restore generalization. Under concept drift (P(Y|X) changes), unlabeled inputs appear identical, rendering the performance drop completely undetectable without new ground-truth labels."
      readouts={
        <>
          <Readout label="deployment accuracy" value={`${results.currentAcc.toFixed(1)}%`} />
          <Readout label="detectable with unlabeled inputs" value={results.detectableUnlabeled} />
          <Readout
            label="unlabeled correction feasible"
            value={state.shiftType === 'concept' ? 'Impossible without labels' : 'Feasible (Importance reweighting)'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Training observations versus deployment test distribution">
        <Points name="training class 0 (blue circles)" x={results.train0X} y={results.train0Y} slot={0} size={4} />
        <Points name="training class 1 (orange circles)" x={results.train1X} y={results.train1Y} slot={1} size={4} />
        <Points
          name="deployment test class 0 (purple squares)"
          x={results.test0X}
          y={results.test0Y}
          slot={2}
          size={5}
        />
        <Points name="deployment test class 1 (red squares)" x={results.test1X} y={results.test1Y} slot={3} size={5} />
        <Curve
          name="nominal training decision boundary"
          x={results.trainBoundaryX}
          y={results.trainBoundaryY}
          slot={0}
          width={2}
        />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ProceduralGenerationVisualizer() {
  const state = useFigureState({
    numTrainingLevels: slider(10, 500, 50, {
      step: 20,
      label: 'training level seed count (K levels)',
    }),
    regularization: choice<'vanilla_impala' | 'augmented_regularized'>(
      [
        { value: 'augmented_regularized', label: 'Data augmentation + L2 weight decay (Crop / Cutout)' },
        { value: 'vanilla_impala', label: 'Vanilla Impala CNN (prone to level memorization)' },
      ],
      'vanilla_impala',
      { label: 'training regularization' },
    ),
  })

  const results = useMemo(() => {
    // Generate generalization curves across level count K in [10, 500]
    const kGrid: number[] = []
    const trainScores: number[] = []
    const testScores: number[] = []

    for (let k = 10; k <= 500; k += 20) {
      kGrid.push(k)

      let train = 0
      let test = 0

      if (state.regularization === 'vanilla_impala') {
        // Vanilla memorizes small K easily
        train = Math.min(9.8, 9.8 - 0.5 * Math.exp(-k / 50))
        // Test generalizes poorly for small K, rises slowly as K -> 500
        test = 9.2 / (1 + Math.exp(-(k - 180) / 75))
      } else {
        // Augmented regularizer reduces training memorization but boosts test generalization
        train = Math.min(9.2, 8.8 + 0.4 * (1 - Math.exp(-k / 80)))
        test = 8.8 / (1 + Math.exp(-(k - 100) / 60))
      }

      trainScores.push(train)
      testScores.push(test)
    }

    // Current point evaluated at selected slider value
    const K = state.numTrainingLevels
    let currentTrain = 0
    let currentTest = 0

    if (state.regularization === 'vanilla_impala') {
      currentTrain = Math.min(9.8, 9.8 - 0.5 * Math.exp(-K / 50))
      currentTest = 9.2 / (1 + Math.exp(-(K - 180) / 75))
    } else {
      currentTrain = Math.min(9.2, 8.8 + 0.4 * (1 - Math.exp(-K / 80)))
      currentTest = 8.8 / (1 + Math.exp(-(K - 100) / 60))
    }

    const genGap = currentTrain - currentTest

    let diagnosis = 'Severe Memorization (Overfitting)'
    if (genGap < 1.0) {
      diagnosis = 'Robust Generalization'
    } else if (genGap < 2.5) {
      diagnosis = 'Moderate Generalization Gap'
    }

    return {
      kGrid,
      trainScores,
      testScores,
      currentK: K,
      currentTrain,
      currentTest,
      genGap,
      diagnosis,
    }
  }, [state.numTrainingLevels, state.regularization])

  const xAxis = useAxis({
    label: 'training set diversity: number of unique procedural seeds K',
    range: [0, 500],
  })
  const yAxis = useAxis({
    label: 'mean benchmark score (max 10.0)',
    range: [0, 10.5],
  })

  return (
    <Figure
      title="Procedural generation in reinforcement learning: training seeds versus generalization gap"
      state={state}
      hoverReadout={false}
      caption="Evaluating the generalization gap in procedurally generated environments (Procgen benchmark). When trained on a small number of levels (K < 50), deep neural policies achieve near-perfect training return by memorizing level-specific obstacle placements and textures, but fail completely on unseen test seeds. Scaling training diversity to K > 200 and applying visual data augmentations forces the policy to learn invariant semantic features, closing the generalization gap."
      readouts={
        <>
          <Readout label="training set score" value={results.currentTrain.toFixed(2)} />
          <Readout label="unseen test score" value={results.currentTest.toFixed(2)} />
          <Readout label="generalization gap (Δ)" value={results.genGap.toFixed(2)} />
          <Readout label="generalization status" value={results.diagnosis} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Training score vs unseen test score as a function of training seeds">
        <Curve
          name="training level performance (memorized)"
          x={results.kGrid}
          y={results.trainScores}
          slot={0}
          width={2.5}
        />
        <Curve
          name="unseen test level performance (generalization)"
          x={results.kGrid}
          y={results.testScores}
          slot={1}
          width={2.5}
        />
        <Points
          name="current operating point (training)"
          x={[results.currentK]}
          y={[results.currentTrain]}
          slot={0}
          size={6}
        />
        <Points
          name="current operating point (test)"
          x={[results.currentK]}
          y={[results.currentTest]}
          slot={1}
          size={6}
        />
      </Plot>
    </Figure>
  )
}

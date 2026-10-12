import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function SubpopulationShiftExplorer() {
  const state = useFigureState({
    trainingObjective: choice<'erm' | 'group_dro'>(
      [
        { value: 'erm', label: 'Standard ERM (Minimizes Average Loss - Learns Shortcut)' },
        { value: 'group_dro', label: 'Group DRO / Re-weighting (Minimizes Worst-Group Loss)' },
      ],
      'erm',
      { label: 'training objective' },
    ),
    spuriousRatio: slider(0.7, 0.99, 0.95, {
      step: 0.02,
      label: 'spurious correlation strength (majority %)',
    }),
  })

  const results = useMemo(() => {
    const rng = stream(73)
    const N_TOTAL = 100

    // Features:
    // X-axis: Spurious background feature (e.g., -1 for Land, +1 for Water)
    // Y-axis: Invariant core anatomy feature (e.g., -1 for Landbird, +1 for Waterbird)
    // Target label: Y > 0 (Waterbird) vs Y < 0 (Landbird)

    const majority0X: number[] = []
    const majority0Y: number[] = []
    const minority0X: number[] = []
    const minority0Y: number[] = []

    const majority1X: number[] = []
    const majority1Y: number[] = []
    const minority1X: number[] = []
    const minority1Y: number[] = []

    const nClass = N_TOTAL / 2
    const nMaj = Math.round(nClass * state.spuriousRatio)
    const nMin = nClass - nMaj

    // Class 0 (Landbirds): Invariant Y ~ -1.2
    for (let i = 0; i < nMaj; i++) {
      // Majority: Land background (X ~ -1.2)
      majority0X.push(normal(rng, -1.2, 0.35))
      majority0Y.push(normal(rng, -1.2, 0.35))
    }
    for (let i = 0; i < nMin; i++) {
      // Minority counterexamples: Water background (X ~ +1.2)
      minority0X.push(normal(rng, 1.2, 0.35))
      minority0Y.push(normal(rng, -1.2, 0.35))
    }

    // Class 1 (Waterbirds): Invariant Y ~ +1.2
    for (let i = 0; i < nMaj; i++) {
      // Majority: Water background (X ~ +1.2)
      majority1X.push(normal(rng, 1.2, 0.35))
      majority1Y.push(normal(rng, 1.2, 0.35))
    }
    for (let i = 0; i < nMin; i++) {
      // Minority counterexamples: Land background (X ~ -1.2)
      minority1X.push(normal(rng, -1.2, 0.35))
      minority1Y.push(normal(rng, 1.2, 0.35))
    }

    // Accuracies
    let averageAcc = 0
    let worstGroupAcc = 0

    if (state.trainingObjective === 'erm') {
      // ERM relies on shortcut X: achieves high average acc by exploiting correlation
      averageAcc = state.spuriousRatio * 100
      // But minority counterexamples fail completely
      worstGroupAcc = (1 - state.spuriousRatio) * 100 * 0.4
    } else {
      // Group DRO focuses on worst group, forcing model to rely on core invariant feature Y
      averageAcc = 93.5
      worstGroupAcc = 91.0
    }

    // Decision boundary line
    // ERM boundary: vertical line X = 0 (splits by background)
    // Group DRO boundary: horizontal line Y = 0 (splits by core anatomy)
    const boundaryX = state.trainingObjective === 'erm' ? [0, 0] : [-2.5, 2.5]
    const boundaryY = state.trainingObjective === 'erm' ? [-2.5, 2.5] : [0, 0]

    return {
      majority0X,
      majority0Y,
      minority0X,
      minority0Y,
      majority1X,
      majority1Y,
      minority1X,
      minority1Y,
      boundaryX,
      boundaryY,
      averageAcc,
      worstGroupAcc,
    }
  }, [state.trainingObjective, state.spuriousRatio])

  const xAxis = useAxis({
    label: 'spurious shortcut feature (background environment)',
    range: [-2.5, 2.5],
  })
  const yAxis = useAxis({
    label: 'invariant core feature (anatomical morphology)',
    range: [-2.5, 2.5],
  })

  return (
    <Figure
      title="Subpopulation shift, shortcut learning, and worst-group risk minimization"
      state={state}
      hoverReadout={false}
      caption="Evaluating models under subpopulation shift and spurious shortcuts. When 95% of landbirds appear on land and 95% of waterbirds appear on water, standard ERM minimizes average loss by adopting a vertical decision boundary on the background shortcut. While average accuracy reaches 95%, worst-group accuracy collapses below 10% on counterexamples (e.g. waterbirds on land). Group DRO penalizes the worst group, forcing the model to discover the invariant horizontal boundary."
      readouts={
        <>
          <Readout label="overall average accuracy" value={`${results.averageAcc.toFixed(1)}%`} />
          <Readout label="worst-group accuracy" value={`${results.worstGroupAcc.toFixed(1)}%`} />
          <Readout
            label="learned mechanism"
            value={
              state.trainingObjective === 'erm' ? 'Spurious Shortcut (Background)' : 'Invariant Feature (Morphology)'
            }
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Feature space partitioned by core and spurious attributes">
        <Points
          name="majority landbird on land (class 0)"
          x={results.majority0X}
          y={results.majority0Y}
          slot={0}
          size={4}
        />
        <Points
          name="minority landbird on water (counterexample)"
          x={results.minority0X}
          y={results.minority0Y}
          slot={1}
          size={5}
        />
        <Points
          name="majority waterbird on water (class 1)"
          x={results.majority1X}
          y={results.majority1Y}
          slot={2}
          size={4}
        />
        <Points
          name="minority waterbird on land (counterexample)"
          x={results.minority1X}
          y={results.minority1Y}
          slot={3}
          size={5}
        />
        <Curve
          name="learned classifier decision boundary"
          x={results.boundaryX}
          y={results.boundaryY}
          slot={0}
          width={2.5}
        />
      </Plot>
    </Figure>
  )
}

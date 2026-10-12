import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LearningCurvesVisualizer() {
  const state = useFigureState({
    curveView: choice<'learning_curves' | 'test_uncertainty'>(
      [
        {
          value: 'learning_curves',
          label: 'Empirical Learning Curves (Train Error vs Validation Error vs Bayes Floor)',
        },
        { value: 'test_uncertainty', label: 'Test Set Sizing & 95% Binomial Margin of Error' },
      ],
      'learning_curves',
      { label: 'sample size perspective' },
    ),
    sampleSizeN: slider(200, 20000, 3000, {
      step: 200,
      label: 'training sample size N',
    }),
    testSetSize: slider(100, 5000, 1000, {
      step: 100,
      label: 'evaluation test set size N_test (confidence interval width)',
    }),
  })

  const results = useMemo(() => {
    const bayesFloor = 4.0 // 4% irreducible Bayes error

    // Generate learning curves across sample sizes [200 .. 20000]
    const nGrid: number[] = []
    const trainCurve: number[] = []
    const valCurve: number[] = []
    const bayesCurve: number[] = []

    for (let n = 200; n <= 20000; n += 400) {
      nGrid.push(n / 1000.0) // k samples
      // Validation error decays: E_val = Bayes + 45.0 * N^(-0.42)
      const eVal = bayesFloor + 45.0 * Math.pow(n / 200.0, -0.42)
      // Training error approaches from below: E_train = Bayes - 3.5 * N^(-0.25)
      const eTrain = Math.max(0.2, bayesFloor - 3.8 * Math.pow(n / 200.0, -0.3))

      valCurve.push(eVal)
      trainCurve.push(eTrain)
      bayesCurve.push(bayesFloor)
    }

    const curN = state.sampleSizeN
    const curValError = bayesFloor + 45.0 * Math.pow(curN / 200.0, -0.42)
    const curTrainError = Math.max(0.2, bayesFloor - 3.8 * Math.pow(curN / 200.0, -0.3))
    const genGap = curValError - curTrainError

    // Test set binomial margin: 1.96 * sqrt(p * (1-p) / N_test)
    const pErr = curValError / 100.0
    const se = Math.sqrt((pErr * (1.0 - pErr)) / state.testSetSize)
    const marginPct = 1.96 * se * 100.0

    // Marginal gain from doubling data: 2 * N
    const nextValError = bayesFloor + 45.0 * Math.pow((2 * curN) / 200.0, -0.42)
    const marginalReduction = curValError - nextValError

    // CI upper/lower bounds
    const valUpper = valCurve.map((v) => v + marginPct)
    const valLower = valCurve.map((v) => Math.max(bayesFloor, v - marginPct))

    return {
      nGrid,
      trainCurve,
      valCurve,
      bayesCurve,
      valUpper,
      valLower,
      curN: curN / 1000.0,
      curValError,
      curTrainError,
      genGap,
      marginPct,
      marginalReduction,
    }
  }, [state.sampleSizeN, state.testSetSize])

  const axisX = useAxis({
    label: 'Training Sample Size N (Thousands of Examples)',
    range: [0.2, 20.2],
  })

  const axisY = useAxis({
    label: 'Error Rate (%)',
    range: [0, 42],
  })

  return (
    <Figure
      title="Learning Curves, Power-Law Scaling and Test Set Uncertainty Bounds"
      state={state}
      hoverReadout={false}
      caption="Validation error decays toward the irreducible Bayes floor following power-law scaling (N^-alpha). The generalization gap shrinks with more training data, while the evaluation test set size dictates whether measured metric improvements are statistically significant or noise."
      readouts={
        <>
          <Readout label="Validation Error" value={`${results.curValError.toFixed(1)}%`} />
          <Readout label="Training Error" value={`${results.curTrainError.toFixed(1)}%`} />
          <Readout label="Generalization Gap" value={`${results.genGap.toFixed(1)}%`} />
          <Readout
            label="Test Set 95% Confidence"
            value={`+/- ${results.marginPct.toFixed(2)}% (N_test = ${state.testSetSize})`}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Empirical Error Trajectory vs Bayes Error Limit">
        <Curve name="Validation Error E_val(N)" x={results.nGrid} y={results.valCurve} slot={0} width={2.5} />
        <Curve name="Training Error E_train(N)" x={results.nGrid} y={results.trainCurve} slot={1} width={2} />
        <Curve
          name="Irreducible Bayes Error Floor"
          x={results.nGrid}
          y={results.bayesCurve}
          slot={2}
          dashed
          width={1.8}
        />
        {state.curveView === 'test_uncertainty' && (
          <>
            <Curve name="95% CI Upper Bound" x={results.nGrid} y={results.valUpper} slot={0} dashed width={1.2} />
            <Curve name="95% CI Lower Bound" x={results.nGrid} y={results.valLower} slot={0} dashed width={1.2} />
          </>
        )}
        <Points name="Operating Sample Size" x={[results.curN]} y={[results.curValError]} slot={0} size={8} />
      </Plot>
    </Figure>
  )
}

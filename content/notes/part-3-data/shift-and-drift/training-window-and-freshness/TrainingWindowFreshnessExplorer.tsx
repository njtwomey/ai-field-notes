import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function TrainingWindowFreshnessExplorer() {
  const state = useFigureState({
    windowLength: slider(10, 80, 35, { step: 5, label: 'training window length (days)' }),
    weightingScheme: choice<'rectangular' | 'exponential'>(
      [
        { value: 'rectangular', label: 'Uniform sliding window (hard cutoff)' },
        { value: 'exponential', label: 'Exponential recency weighting (gamma = 0.94)' },
      ],
      'rectangular',
      { label: 'window weighting scheme' },
    ),
    regimeBreak: choice<'none' | 'structural_break'>(
      [
        { value: 'none', label: 'Smooth gradual drift' },
        { value: 'structural_break', label: 'Sudden regime shift at t = 50' },
      ],
      'structural_break',
      { label: 'temporal regime dynamic' },
    ),
  })

  const results = useMemo(() => {
    const rng = stream(61)
    const T_TOTAL = 80
    const timeIndices: number[] = []
    const trueSignal: number[] = []
    const noisyObservations: number[] = []

    for (let t = 0; t <= T_TOTAL; t++) {
      timeIndices.push(t)
      // Base non-stationary dynamic: gradual linear slope
      let sig = 0.03 * t
      if (state.regimeBreak === 'structural_break' && t >= 50) {
        // Sudden downward regime shift at t = 50
        sig -= 1.8
      }
      trueSignal.push(sig)
      noisyObservations.push(sig + normal(rng, 0, 0.45))
    }

    // Deploy model at current time T = 80 to predict immediate next step T = 81
    const T_NOW = 80
    const W = state.windowLength
    const windowStart = Math.max(0, T_NOW - W)

    // Fit model using selected window
    let weightSum = 0
    let weightedTargetSum = 0
    const windowPointsX: number[] = []
    const windowPointsY: number[] = []

    for (let t = windowStart; t <= T_NOW; t++) {
      let w = 1.0
      if (state.weightingScheme === 'exponential') {
        w = Math.pow(0.94, T_NOW - t)
      }
      weightSum += w
      weightedTargetSum += w * noisyObservations[t]
      windowPointsX.push(t)
      windowPointsY.push(noisyObservations[t])
    }

    const currentPrediction = weightSum > 0 ? weightedTargetSum / weightSum : 0
    const trueFutureVal = trueSignal[T_NOW]

    // Bias = difference between expected prediction and true current state
    // Variance = inversely proportional to effective sample size
    const effSampleSize = state.weightingScheme === 'rectangular' ? W : weightSum * 0.75
    const variance = 0.45 ** 2 / Math.max(1, effSampleSize)
    const bias = Math.abs(currentPrediction - trueFutureVal)
    const mse = bias * bias + variance

    let diagnosis = 'Balanced Freshness'
    if (W < 20) {
      diagnosis = 'High Variance (Window too short / noisy)'
    } else if (
      state.regimeBreak === 'structural_break' &&
      windowStart < 50 &&
      state.weightingScheme === 'rectangular'
    ) {
      diagnosis = 'Severe Stale Bias (Includes pre-break regime)'
    } else if (W > 55 && state.weightingScheme === 'rectangular') {
      diagnosis = 'Stale Bias (Lagging drift)'
    }

    // Prediction horizontal line across window
    const predLineX = [windowStart, T_NOW]
    const predLineY = [currentPrediction, currentPrediction]

    return {
      timeIndices,
      trueSignal,
      noisyObservations,
      windowPointsX,
      windowPointsY,
      predLineX,
      predLineY,
      currentPrediction,
      bias,
      variance,
      mse,
      diagnosis,
    }
  }, [state.windowLength, state.weightingScheme, state.regimeBreak])

  const xAxis = useAxis({
    label: 'historical time index t (current time T = 80)',
    range: [0, 80],
  })
  const yAxis = useAxis({
    label: 'time series value / prediction',
    range: [-2.5, 3.0],
  })

  return (
    <Figure
      title="Training window length and data freshness: the temporal bias-variance trade-off"
      state={state}
      hoverReadout={false}
      caption="Evaluating training window length under temporal distribution shifts. A short window (W < 20) minimizes stale data bias but suffers high estimation variance from small sample size. A long window (W > 60) pools abundant samples but introduces stale historical bias, especially across structural regime shifts. Exponential recency weighting discounts obsolete history, stabilizing predictions across regime breaks."
      readouts={
        <>
          <Readout label="current model prediction" value={results.currentPrediction.toFixed(2)} />
          <Readout label="historical bias" value={results.bias.toFixed(3)} />
          <Readout label="sample variance" value={results.variance.toFixed(3)} />
          <Readout label="total generalization error (MSE)" value={results.mse.toFixed(3)} />
          <Readout label="freshness balance" value={results.diagnosis} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Historical telemetry, structural breaks, and windowed estimator">
        <Curve
          name="underlying non-stationary target dynamic"
          x={results.timeIndices}
          y={results.trueSignal}
          slot={0}
          width={2}
        />
        <Points
          name="historical noisy observations"
          x={results.timeIndices}
          y={results.noisyObservations}
          slot={2}
          size={3}
        />
        <Points
          name="active training window observations"
          x={results.windowPointsX}
          y={results.windowPointsY}
          slot={1}
          size={5}
        />
        <Curve name="window-fitted model prediction" x={results.predLineX} y={results.predLineY} slot={3} width={3} />
      </Plot>
    </Figure>
  )
}

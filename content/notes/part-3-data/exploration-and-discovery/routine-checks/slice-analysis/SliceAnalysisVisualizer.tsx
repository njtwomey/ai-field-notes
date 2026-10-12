import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SliceAnalysisVisualizer() {
  const state = useFigureState({
    analysisMode: choice<'disparity' | 'shrinkage' | 'simpsons'>(
      [
        { value: 'disparity', label: 'Disaggregated Error Rates (Aggregate vs Subgroups)' },
        { value: 'shrinkage', label: 'Thin Slices & Empirical Bayes Shrinkage' },
        { value: 'simpsons', label: 'Simpson’s Paradox (Aggregate Inversion)' },
      ],
      'disparity',
      { label: 'slice analysis lens' },
    ),
    thinSampleSize: slider(5, 80, 15, {
      step: 5,
      label: 'thin subgroup sample size (N_c)',
    }),
    intersectionalDifficulty: slider(1.0, 3.5, 2.4, {
      step: 0.2,
      label: 'minority subgroup relative difficulty multiplier',
    }),
  })

  const { analysisMode, thinSampleSize, intersectionalDifficulty } = state

  const results = useMemo(() => {
    if (analysisMode === 'disparity') {
      // 4 slices:
      // S1: Majority Tier 1 (N = 8000, error = 4.2%)
      // S2: Majority Tier 2 (N = 5000, error = 6.1%)
      // S3: Thin Enterprise (N = 300, error = 12.0%)
      // S4: Intersectional Minority (N = 200, error = 4.2 * difficulty)
      const slices = [1, 2, 3, 4]
      const s4Error = Math.min(65.0, 4.2 * intersectionalDifficulty * 3.5)
      const errorRates = [4.2, 6.1, 12.0, s4Error]
      // Weighted aggregate: (8000*4.2 + 5000*6.1 + 300*12 + 200*s4Error) / 13500
      const aggError = (8000 * 4.2 + 5000 * 6.1 + 300 * 12 + 200 * s4Error) / 13500
      const aggregateLine = slices.map(() => aggError)

      return {
        x: slices,
        y: errorRates,
        benchmark: aggregateLine,
        aggError,
        s4Error,
        gap: s4Error - aggError,
      }
    } else if (analysisMode === 'shrinkage') {
      // Show shrinkage across sample sizes N from 5 to 100
      const nValues = [5, 10, 15, 25, 40, 60, 80, 100]
      const priorMean = 8.0 // global error 8%
      const rawObserved = 24.0 // raw observed error in slice

      // Shrinkage formula: lambda = N / (N + tau^2)
      // shrunk = lambda * raw + (1 - lambda) * prior
      const tau = 25.0
      const shrunkEstimates = nValues.map((n) => {
        const weight = n / (n + tau)
        return weight * rawObserved + (1 - weight) * priorMean
      })

      const rawLine = nValues.map(() => rawObserved)
      const priorLine = nValues.map(() => priorMean)

      const currentWeight = thinSampleSize / (thinSampleSize + tau)
      const currentShrunk = currentWeight * rawObserved + (1 - currentWeight) * priorMean

      return {
        x: nValues,
        y: shrunkEstimates,
        benchmark: rawLine,
        priorLine,
        aggError: priorMean,
        s4Error: currentShrunk,
        gap: rawObserved - currentShrunk,
      }
    } else {
      // Simpson's paradox: 2 strata (Stratum 1 = Simple, Stratum 2 = Complex)
      // Group A (Slice 1): Stratum 1 rate 15%, Stratum 2 rate 40%. Aggregate 18% (80% simple)
      // Group B (Slice 2): Stratum 1 rate 25%, Stratum 2 rate 50%. Aggregate 45% (80% complex)
      const xIdx = [1, 2]
      // Aggregate rates
      const aggRates = [18.0, 45.0]
      // Within Stratum 1 (Simple cases): Group A = 15%, Group B = 22%
      // Within Stratum 2 (Complex cases): Group A = 38%, Group B = 48%
      const stratum1 = [15.0, 22.0]

      return {
        x: xIdx,
        y: aggRates,
        benchmark: stratum1,
        aggError: 18.0,
        s4Error: 45.0,
        gap: 27.0,
      }
    }
  }, [analysisMode, thinSampleSize, intersectionalDifficulty])

  const axisX = useAxis({
    label:
      analysisMode === 'disparity'
        ? 'Data Slices (1: Majority A, 2: Majority B, 3: Small Enterprise, 4: Intersectional Cohort)'
        : analysisMode === 'shrinkage'
          ? 'Subgroup Sample Size (N)'
          : 'Demographic Group (1: Group Alpha, 2: Group Beta)',
    range: analysisMode === 'disparity' ? [0.5, 4.5] : analysisMode === 'shrinkage' ? [0, 105] : [0.5, 2.5],
  })

  const axisY = useAxis({
    label:
      analysisMode === 'disparity'
        ? 'Error Rate (%)'
        : analysisMode === 'shrinkage'
          ? 'Estimated Subgroup Error Rate (%)'
          : 'Positive Outcome Rate (%)',
    range: [0, 60],
  })

  return (
    <Figure
      title="Disaggregated Subgroup and Slice Analysis Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating models across disaggregated data slices exposes masked disparities where strong aggregate metrics conceal severe error rates in intersectional minority cohorts and thin slices."
      readouts={
        <>
          {analysisMode === 'disparity' && (
            <>
              <Readout label="Aggregate Error Rate" value={`${results.aggError.toFixed(1)}%`} />
              <Readout label="Intersectional Slice Error (S4)" value={`${results.s4Error.toFixed(1)}%`} />
              <Readout
                label="Disparity Gap"
                value={`+${results.gap.toFixed(1)}% pts`}
                color={results.gap > 15 ? 'var(--red-600)' : 'var(--blue-600)'}
              />
            </>
          )}
          {analysisMode === 'shrinkage' && (
            <>
              <Readout label="Raw Slice Mean (N=15)" value="24.0% (Uncertain)" />
              <Readout label="Empirical Bayes Shrunk Mean" value={`${results.s4Error.toFixed(1)}%`} />
              <Readout label="Shrinkage toward Prior" value={`-${results.gap.toFixed(1)}% pts`} />
            </>
          )}
          {analysisMode === 'simpsons' && (
            <>
              <Readout label="Aggregate Group Alpha" value="18.0% positive" />
              <Readout label="Aggregate Group Beta" value="45.0% positive" />
              <Readout label="Paradox Status" value="Inversion caused by unequal case complexity mix" />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Slice Analysis Profile">
        <Curve
          name={analysisMode === 'disparity' ? 'Global Aggregate Error' : 'Baseline / Stratum Reference'}
          x={results.x}
          y={results.benchmark}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Disaggregated Slice Metrics"
          x={results.x}
          y={results.y}
          slot={results.gap > 15 && analysisMode === 'disparity' ? 1 : 0}
          size={8}
        />
        {analysisMode === 'shrinkage' && (
          <Curve name="Empirical Bayes Shrinkage Curve" x={results.x} y={results.y} slot={0} width={2.5} />
        )}
      </Plot>
    </Figure>
  )
}

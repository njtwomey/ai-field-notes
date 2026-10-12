import { useMemo } from 'react'
import { Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DeletionMethodsExplorer() {
  const state = useFigureState({
    missingRate: slider(0.01, 0.2, 0.05, { step: 0.01, label: 'missing per feature p' }),
    numFeatures: int(20, { min: 2, max: 60, label: 'feature count D' }),
    totalRows: int(1000, { min: 100, max: 5000, label: 'sample size N' }),
  })

  const p = state.missingRate
  const D = state.numFeatures
  const N = state.totalRows

  // Analytical retention: P(complete) = (1 - p)^D under independent missingness
  const retentionFraction = Math.pow(1 - p, D)
  const retainedRows = Math.round(N * retentionFraction)
  const lostFraction = 1 - retentionFraction

  // Pairwise retention: P(both j and k observed) = (1 - p)^2
  const pairwiseFraction = Math.pow(1 - p, 2)
  const pairwiseRows = Math.round(N * pairwiseFraction)

  // Generate retention curves for D = 1 to 60 for different missingness rates: 2%, 5%, 10%, and current p
  const dRange = useMemo(() => Array.from({ length: 60 }, (_, i) => i + 1), [])

  const currentCurveY = useMemo(() => {
    return dRange.map((d) => Math.pow(1 - p, d) * 100)
  }, [dRange, p])

  const benchmarkCurve2 = useMemo(() => {
    return dRange.map((d) => Math.pow(1 - 0.02, d) * 100)
  }, [dRange])

  const benchmarkCurve10 = useMemo(() => {
    return dRange.map((d) => Math.pow(1 - 0.1, d) * 100)
  }, [dRange])

  // Single point marker for current selection
  const markerX = [D]
  const markerY = [retentionFraction * 100]

  const xAxis = useAxis({ label: 'feature dimension D', range: [1, 60] })
  const yAxis = useAxis({ label: 'complete rows retained (%)', range: [0, 100] })

  return (
    <Figure
      title="Exponential row loss in complete-case analysis"
      state={state}
      hoverReadout={false}
      caption="Under independent missingness across features with per-column rate p, the probability of retaining a complete row decays exponentially as (1 − p)^D. Even with a modest 5% missingness rate, scaling to D = 40 features discards over 87% of observations. Pairwise deletion preserves (1 − p)^2 ≈ 90% of records for each bivariate cell, but risks producing non-positive semi-definite covariance matrices."
      readouts={
        <>
          <Readout label="complete cases N_cc" value={`${retainedRows} (${(retentionFraction * 100).toFixed(1)}%)`} />
          <Readout label="discarded rows" value={`${N - retainedRows} (${(lostFraction * 100).toFixed(1)}%)`} />
          <Readout label="pairwise cell N_jk" value={`${pairwiseRows} (${(pairwiseFraction * 100).toFixed(1)}%)`} />
          <Readout
            label="relative efficiency N_cc / N_jk"
            value={`${(retentionFraction / pairwiseFraction).toFixed(3)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Complete-case retention vs feature dimensionality">
        <Curve name="benchmark p = 2% per column" x={dRange} y={benchmarkCurve2} slot={1} dashed width={1.5} />
        <Curve
          name={`current rate p = ${(p * 100).toFixed(0)}% per column`}
          x={dRange}
          y={currentCurveY}
          slot={0}
          width={2.5}
        />
        <Curve name="benchmark p = 10% per column" x={dRange} y={benchmarkCurve10} slot={2} dashed width={1.5} />
        <Points name="current configuration" x={markerX} y={markerY} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

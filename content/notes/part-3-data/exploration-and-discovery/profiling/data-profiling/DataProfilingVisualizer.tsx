import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataProfilingVisualizer() {
  const state = useFigureState({
    profileMode: choice<'cardinality' | 'missingness' | 'outliers'>(
      [
        { value: 'cardinality', label: 'Column Cardinality & Distinct Ratios' },
        { value: 'missingness', label: 'Missingness & Null Ingestion Patterns' },
        { value: 'outliers', label: 'Tukey Outlier Fences & Skewness' },
      ],
      'cardinality',
      { label: 'profiling inspection lens' },
    ),
    missingSeverity: slider(0, 80, 25, {
      step: 5,
      label: 'simulated missingness rate (%)',
    }),
    outlierSeverity: slider(1.0, 5.0, 2.5, {
      step: 0.5,
      label: 'extreme tail outlier multiplier',
    }),
  })

  const { profileMode, missingSeverity, outlierSeverity } = state

  const results = useMemo(() => {
    if (profileMode === 'cardinality') {
      // 6 columns:
      // 1: id (100% distinct, PK)
      // 2: os (0.05% distinct, Low-cardinality)
      // 3: zip (22% distinct, High-cardinality)
      // 4: currency (0.001% distinct, Quasi-constant)
      // 5: amount (82% distinct, Continuous)
      // 6: target (0.002% distinct, Binary)
      const cols = [1, 2, 3, 4, 5, 6]
      const distinctPct = [100.0, 0.5, 22.0, 0.01, 82.0, 0.02]
      const quasiConstantThresh = cols.map(() => 0.1)

      return {
        x: cols,
        y: distinctPct,
        baseline: quasiConstantThresh,
        status: 'Audit: Column 4 is quasi-constant (drop); Column 3 is high-cardinality',
      }
    } else if (profileMode === 'missingness') {
      const cols = [1, 2, 3, 4, 5, 6]
      // Missingness rates across columns
      // Col 1 (ID): 0%
      // Col 2 (OS): 1.5%
      // Col 3 (Zip): missingSeverity * 0.4
      // Col 4 (Currency): 0%
      // Col 5 (Amount): missingSeverity * 0.8
      // Col 6 (Target): 0%
      const missRates = [0, 1.5, missingSeverity * 0.4, 0, missingSeverity * 0.8, 0]
      const criticalThresh = cols.map(() => 30.0)

      return {
        x: cols,
        y: missRates,
        baseline: criticalThresh,
        status:
          missingSeverity > 40
            ? 'High missingness in Col 5 (>30%): requires MNAR check'
            : 'Missingness within standard imputation limits',
      }
    } else {
      // Outlier box-plot quantile points for Col 5 (Amount) across 7 quantiles
      // p1, p10, p25 (Q1), p50 (Median), p75 (Q3), p99, p99.9 (Outlier)
      const qIdx = [1, 2, 3, 4, 5, 6, 7]
      const qValues = [1.2, 8.5, 22.0, 45.0, 92.0, 220.0, 220.0 * outlierSeverity]
      // Tukey Upper Fence: Q3 + 1.5 * IQR = 92 + 1.5 * (92 - 22) = 197
      const upperFence = qIdx.map(() => 197.0)

      return {
        x: qIdx,
        y: qValues,
        baseline: upperFence,
        status:
          outlierSeverity > 2.0
            ? `Heavy tail: Max value is ${(220 * outlierSeverity).toFixed(0)} (Tukey fence is 197)`
            : 'Moderate tail within fences',
      }
    }
  }, [profileMode, missingSeverity, outlierSeverity])

  const axisX = useAxis({
    label:
      profileMode === 'outliers'
        ? 'Distribution Quantile Index (1: Min, 3: Q1, 4: Median, 5: Q3, 6: P99, 7: Extreme Max)'
        : 'Dataset Columns (1: ID, 2: OS, 3: PostalCode, 4: Currency, 5: Amount, 6: Label)',
    range: profileMode === 'outliers' ? [0.5, 7.5] : [0.5, 6.5],
  })

  const axisY = useAxis({
    label:
      profileMode === 'cardinality'
        ? 'Uniqueness / Distinct Ratio (%)'
        : profileMode === 'missingness'
          ? 'Missing / Null Percentage (%)'
          : 'Feature Value ($ / Scaled Units)',
    range: profileMode === 'cardinality' ? [0, 105] : profileMode === 'missingness' ? [0, 85] : [0, 1200],
  })

  return (
    <Figure
      title="Automated Data Profiling Diagnostic Explorer"
      state={state}
      hoverReadout={false}
      caption="Automated data profiling audits column metadata, uniqueness ratios, missingness matrices, and Tukey outlier fences to flag structural issues before feature engineering."
      readouts={
        <>
          {profileMode === 'cardinality' && (
            <>
              <Readout label="Primary Key Candidate" value="Column 1 (100% distinct, 0 nulls)" />
              <Readout label="High-Cardinality Warning" value="Column 3 (22% unique, requires encoding)" />
              <Readout label="Quasi-Constant Flag" value="Column 4 (<0.05% unique, safe to drop)" />
            </>
          )}
          {profileMode === 'missingness' && (
            <>
              <Readout label="Max Column Missingness" value={`${(missingSeverity * 0.8).toFixed(1)}% (Col 5)`} />
              <Readout label="Integrity Status" value={results.status} />
            </>
          )}
          {profileMode === 'outliers' && (
            <>
              <Readout label="Tukey Upper Fence" value="197.0 (Q3 + 1.5 IQR)" />
              <Readout label="Extreme Max Outlier" value={`${(220 * outlierSeverity).toFixed(0)} units`} />
              <Readout
                label="Diagnostic Action"
                value={outlierSeverity > 2.0 ? 'Apply log1p or winsorization' : 'Within normal variance'}
              />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Column Profile Diagnostics">
        <Curve
          name="Diagnostic Baseline / Critical Fence"
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Points
          name="Observed Column Metrics"
          x={results.x}
          y={results.y}
          slot={profileMode === 'cardinality' && results.y[2] > 20 ? 1 : 0}
          size={8}
        />
        {profileMode === 'outliers' && (
          <Curve name="Distribution Quantile Profile" x={results.x} y={results.y} slot={0} width={2.2} />
        )}
      </Plot>
    </Figure>
  )
}

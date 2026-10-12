import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TemporalProfilingVisualizer() {
  const state = useFigureState({
    diagnosticView: choice<'volume' | 'step_drift' | 'file_order'>(
      [
        { value: 'volume', label: 'Volume & Outages (Gaps, Weekend Dips, Backfill Spikes)' },
        { value: 'step_drift', label: 'Feature Step Drift (Distributional Jumps Across Days)' },
        { value: 'file_order', label: 'File Order Artifacts (Accidental Sorting & Clumping)' },
      ],
      'volume',
      { label: 'temporal profiling diagnostic' },
    ),
    outageLength: slider(0, 4, 2, {
      step: 1,
      label: 'pipeline outage length (days)',
    }),
    stepShift: slider(0.0, 3.0, 1.8, {
      step: 0.2,
      label: 'sensor recalibration shift magnitude (sigma)',
    }),
    sortSeverity: slider(0.0, 1.0, 0.8, {
      step: 0.1,
      label: 'file sorting correlation with target',
    }),
  })

  const { diagnosticView, outageLength, stepShift, sortSeverity } = state

  const results = useMemo(() => {
    if (diagnosticView === 'volume') {
      const days = Array.from({ length: 30 }, (_, i) => i + 1)
      const baseDaily = 1000
      const volumes = days.map((d) => {
        // Day of week seasonality: day 6, 7, 13, 14, 20, 21, 27, 28 are weekends
        const isWeekend = d % 7 === 6 || d % 7 === 0
        let vol = isWeekend ? baseDaily * 0.45 : baseDaily

        // Outage window starting at day 15
        if (outageLength > 0 && d >= 15 && d < 15 + outageLength) {
          vol = 0
        } else if (outageLength > 0 && d === 15 + outageLength) {
          // Catch-up backfill burst
          vol += outageLength * baseDaily * 0.9
        }
        return vol
      })

      const expected = days.map((d) => (d % 7 === 6 || d % 7 === 0 ? baseDaily * 0.45 : baseDaily))

      return {
        x: days,
        y: volumes,
        baseline: expected,
        alert: outageLength > 0 ? `Outage: ${outageLength} day(s) missing records` : 'Nominal ingestion stream',
      }
    } else if (diagnosticView === 'step_drift') {
      const days = Array.from({ length: 30 }, (_, i) => i + 1)
      const baseMean = 10.0
      const means = days.map((d) => {
        // Step jump occurs at day 16 (e.g. app update / sensor swap)
        const shift = d >= 16 ? stepShift * 3.5 : 0
        const noise = Math.sin(d * 1.5) * 0.4
        return baseMean + shift + noise
      })

      const preMeanLine = days.map(() => baseMean)

      return {
        x: days,
        y: means,
        baseline: preMeanLine,
        alert:
          stepShift > 1.0
            ? `Significant step shift at Day 16 (+${(stepShift * 3.5).toFixed(1)} units)`
            : 'Stable rolling mean',
      }
    } else {
      // file_order: 100 row index buckets across the raw file
      const buckets = Array.from({ length: 30 }, (_, i) => i + 1)
      const targetRate = buckets.map((b) => {
        // If sortSeverity = 0, flat 20% target rate
        // If sortSeverity = 1, file begins with 0% target and surges to 80% at end
        const norm = (b - 1) / 29
        const sortedTrend = norm * 60
        const mixed = 20 * (1 - sortSeverity) + sortedTrend * sortSeverity
        return Math.max(0, Math.min(100, mixed))
      })

      const neutral = buckets.map(() => 20)

      return {
        x: buckets,
        y: targetRate,
        baseline: neutral,
        alert:
          sortSeverity > 0.4
            ? 'CRITICAL: File ordered by target/time! Sequential split invalid'
            : 'Random row ordering',
      }
    }
  }, [diagnosticView, outageLength, stepShift, sortSeverity])

  const axisX = useAxis({
    label:
      diagnosticView === 'file_order'
        ? 'Physical File Position (Row Percentile / Batch)'
        : 'Chronological Day Index (1 to 30)',
    range: [0.5, 30.5],
  })

  const axisY = useAxis({
    label:
      diagnosticView === 'volume'
        ? 'Daily Ingested Row Count'
        : diagnosticView === 'step_drift'
          ? 'Feature Mean (Tracking Window)'
          : 'Positive Target Proportion (%)',
    range: diagnosticView === 'volume' ? [0, 4500] : diagnosticView === 'step_drift' ? [5, 25] : [0, 80],
  })

  return (
    <Figure
      title="Temporal and Order Profiling Diagnostic Explorer"
      state={state}
      hoverReadout={false}
      caption="Profiling data across chronological timestamps and physical file index detects pipeline outages, late-arriving backfills, sudden feature shifts, and catastrophic row ordering artifacts."
      readouts={
        <>
          {diagnosticView === 'volume' && (
            <>
              <Readout label="Pipeline Status" value={results.alert} />
              <Readout
                label="Backfill Spike"
                value={outageLength > 0 ? `+${(outageLength * 900).toFixed(0)} rows queued` : 'Smooth ingestion'}
              />
            </>
          )}
          {diagnosticView === 'step_drift' && (
            <>
              <Readout label="Drift Severity" value={results.alert} />
              <Readout
                label="Recalibration Impact"
                value={stepShift > 1.0 ? 'Requires regime segmentation or normalization' : 'Within normal variance'}
              />
            </>
          )}
          {diagnosticView === 'file_order' && (
            <>
              <Readout label="Ordering Risk" value={results.alert} />
              <Readout
                label="Split Safety"
                value={
                  sortSeverity > 0.4
                    ? 'Sequential split causes massive train/test covariate shift'
                    : 'Safe for splitting'
                }
              />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Temporal Profile Diagnostics">
        <Curve name="Nominal Baseline" x={results.x} y={results.baseline} slot={2} dashed width={1.5} />
        <Curve name="Observed Profile" x={results.x} y={results.y} slot={0} width={2.5} />
        <Points
          name="Daily Samples"
          x={results.x}
          y={results.y}
          slot={diagnosticView === 'file_order' && sortSeverity > 0.4 ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}

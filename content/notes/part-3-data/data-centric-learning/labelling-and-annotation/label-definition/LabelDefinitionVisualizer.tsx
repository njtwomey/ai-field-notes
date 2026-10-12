import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LabelDefinitionVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'maturity' | 'censoring_trap' | 'thresholds'>(
      [
        { value: 'maturity', label: 'Cumulative Label Maturity (Window Length vs Completeness)' },
        { value: 'censoring_trap', label: 'Immature Cohort Truncation (Exclusions vs Artificial Depression)' },
        { value: 'thresholds', label: 'Definition Sensitivity (Delinquency Stringency vs Base Rate)' },
      ],
      'maturity',
      { label: 'label definition diagnostic' },
    ),
    windowDays: slider(7, 90, 30, {
      step: 7,
      label: 'outcome window observation horizon (days)',
    }),
    filterImmature: choice<'yes' | 'no'>(
      [
        { value: 'yes', label: 'Filter Immature Cohorts (Clean Training Set)' },
        { value: 'no', label: 'Keep Immature Cohorts (Severe Zero-Label Bias)' },
      ],
      'yes',
      { label: 'immature cohort filtering' },
    ),
  })

  const { visualizerMode, windowDays, filterImmature } = state

  const results = useMemo(() => {
    if (visualizerMode === 'maturity') {
      // Days from prediction: 1 to 90
      const days = [1, 3, 7, 14, 21, 30, 45, 60, 75, 90]
      // Lognormal cumulative conversion curve reaching 15% asymptotically
      const maxRate = 16.0
      const cumRates = days.map((d) => {
        // e.g. CDF of lognormal with median ~ 12 days
        const frac = 1 - Math.exp(-Math.pow(d / 18, 0.75))
        return maxRate * frac
      })

      // Completeness at chosen windowDays
      const chosenFrac = 1 - Math.exp(-Math.pow(windowDays / 18, 0.75))
      const capturedRate = maxRate * chosenFrac
      const completenessPct = (chosenFrac * 100).toFixed(1)

      const cutoffLine = days.map((d) => (d === windowDays ? maxRate : 0))

      return {
        x: days,
        y: cumRates,
        baseline: cutoffLine,
        stat1: `${capturedRate.toFixed(1)}% positive rate`,
        stat2: `${completenessPct}% of lifetime conversions captured`,
        warning:
          windowDays < 21 ? 'Short window misses >30% of slow-maturing positive events' : 'Balanced maturity capture',
      }
    } else if (visualizerMode === 'censoring_trap') {
      // 12 monthly cohorts: month 1 (oldest, fully mature) to month 12 (today, completely immature)
      const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      const trueBase = 15.0

      const rates = months.map((m) => {
        // Older months (1 to 8) are fully mature (> 90 days old)
        if (m <= 8) return trueBase
        // Recent months: if filterImmature === 'yes', excluded (or flat trueBase)
        // If filterImmature === 'no', uncompleted window rows are naively treated as 0s!
        if (filterImmature === 'yes') {
          return trueBase
        } else {
          // Sharp depression toward 0 in months 9 to 12
          const ageMonths = 12 - m
          return trueBase * (ageMonths / 4.0)
        }
      })

      const expected = months.map(() => trueBase)

      return {
        x: months,
        y: rates,
        baseline: expected,
        stat1: filterImmature === 'yes' ? 'Immature cohorts masked' : 'Unfiltered recent data included',
        stat2: filterImmature === 'yes' ? '15.0% uncorrupted rate' : 'Depressed down to 1.2% in Month 12',
        warning:
          filterImmature === 'no'
            ? 'CRITICAL TRAP: Uncompleted outcome windows falsely labeled as negatives (0)'
            : 'Correctly quarantined trailing edge',
      }
    } else {
      // Stringency thresholds:
      // T1: 7-day late payment (18.5% rate)
      // T2: 30-day delinquency (11.2% rate)
      // T3: 60-day delinquency (5.4% rate)
      // T4: 90-day formal default (1.8% rate)
      // T5: 180-day charge-off (0.6% rate)
      const thresholds = [1, 2, 3, 4, 5]
      const rates = [18.5, 11.2, 5.4, 1.8, 0.6]
      const neutral = thresholds.map(() => 5.0)

      return {
        x: thresholds,
        y: rates,
        baseline: neutral,
        stat1: 'Base rate ranges from 18.5% to 0.6%',
        stat2: 'Class imbalance shifts from 5:1 to 160:1',
        warning: 'Align label stringency with intervention economics, not arbitrary calendar thresholds',
      }
    }
  }, [visualizerMode, windowDays, filterImmature])

  const axisX = useAxis({
    label:
      visualizerMode === 'maturity'
        ? 'Observation Elapsed Days After Event (t_pred)'
        : visualizerMode === 'censoring_trap'
          ? 'Historical Monthly Ingestion Cohort (1: Oldest, 12: Current Month)'
          : 'Outcome Definition Stringency (1: 7d Late, 2: 30d, 3: 60d, 4: 90d Default, 5: Charge-Off)',
    range: visualizerMode === 'maturity' ? [0, 95] : visualizerMode === 'censoring_trap' ? [0.5, 12.5] : [0.5, 5.5],
  })

  const axisY = useAxis({
    label: 'Observed Positive Target Rate (%)',
    range: [0, 22],
  })

  return (
    <Figure
      title="Label Definition & Outcome Horizon Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating label definition mechanics: Cumulative conversion maturity trade-offs, the severe risk of immature cohort label depression, and definition sensitivity across delinquency stringency thresholds."
      readouts={
        <>
          <Readout label="Definition Metric" value={results.stat1} />
          <Readout label="Target State" value={results.stat2} />
          <Readout
            label="Operational Warning"
            value={results.warning}
            color={
              visualizerMode === 'censoring_trap' && filterImmature === 'no' ? 'var(--red-600)' : 'var(--blue-600)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Label Definition Trajectory">
        <Curve name="Nominal / Uncorrupted Trajectory" x={results.x} y={results.baseline} slot={2} dashed width={1.8} />
        <Curve name="Observed Positive Prevalence" x={results.x} y={results.y} slot={0} width={2.5} />
        <Points
          name="Observation Points"
          x={results.x}
          y={results.y}
          slot={visualizerMode === 'censoring_trap' && filterImmature === 'no' ? 1 : 0}
          size={7}
        />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function PanelDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'within_vs_between' | 'attrition_dynamics' | 'splitting_paradigms'>(
      [
        {
          value: 'within_vs_between',
          label: 'Variance Decomposition: Within vs Between Unit Variation (Wooldridge 2010)',
        },
        { value: 'attrition_dynamics', label: 'Unbalanced Panels: Non-Random Cohort Attrition & Missingness' },
        { value: 'splitting_paradigms', label: 'Splitting Strategy: Entity Group vs Temporal Cutoff Generalization' },
      ],
      'within_vs_between',
      { label: 'panel analysis perspective' },
    ),
    entityCount: slider(50, 500, 200, {
      step: 25,
      label: 'cross-sectional entities N',
    }),
    timePeriods: slider(2, 12, 6, {
      step: 1,
      label: 'repeated observation periods T',
    }),
  })

  const { visualizerMode, entityCount, timePeriods } = state

  const results = useMemo(() => {
    if (visualizerMode === 'within_vs_between') {
      // Intraclass correlation coefficient (ICC) rho = sigma_between^2 / (sigma_between^2 + sigma_within^2)
      // Range of ICC from 0.1 to 0.9 across 9 points
      const iccs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      // Between-unit variance share (%)
      const betweenShare = iccs.map((rho) => rho * 100)
      // Within-unit variance share (%)
      const withinShare = iccs.map((rho) => (1 - rho) * 100)

      const currICC = 0.55
      const currBetween = currICC * 100
      const currWithin = (1 - currICC) * 100

      return {
        x: iccs.map((v) => v * 100),
        y: betweenShare,
        baseline: withinShare,
        stat1: `Entities N = ${entityCount}, T = ${timePeriods} periods`,
        stat2: `Between-unit variance: ${currBetween.toFixed(0)}%`,
        stat3: `Within-unit variance: ${currWithin.toFixed(0)}%`,
        warning:
          currBetween > 60
            ? 'Dominant unobserved heterogeneity: Cross-sectional entity variance dwarfs longitudinal change; naive pooled models overfit entity identity'
            : 'Balanced variance: Longitudinal within-unit trajectories provide informative signal for fixed-effects modeling',
        axisXLabel: 'Intraclass Correlation Index (ICC %)',
        axisYLabel: 'Between-Entity Variance Fraction (%)',
        xRange: [5, 95] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (visualizerMode === 'attrition_dynamics') {
      // Attrition trajectory across periods t = 1 to T
      const periods = [1, 2, 3, 4, 5, 6, 8, 10, 12]
      // Dropout rate per period (e.g. 8% attrition per period)
      const retentionClean = periods.map(() => 100)
      const retentionObserved = periods.map((t) => Math.round(100 * Math.pow(0.88, t - 1)))

      const currRetention = Math.round(100 * Math.pow(0.88, timePeriods - 1))
      const survivingEntities = Math.round((entityCount * currRetention) / 100)

      return {
        x: periods,
        y: retentionObserved,
        baseline: retentionClean,
        stat1: `Initial cohort N = ${entityCount}`,
        stat2: `Period ${timePeriods} retention: ${currRetention}%`,
        stat3: `Surviving units: ${survivingEntities}`,
        warning:
          currRetention < 60
            ? 'Severe panel attrition: Over 40% of cohort dropped out; survival selection bias distorts long-term outcome predictions'
            : 'Moderate attrition: Panel remains relatively balanced, but missingness mechanisms must be verified',
        axisXLabel: 'Panel Observation Period Index (t)',
        axisYLabel: 'Cohort Active Retention Rate (%)',
        xRange: [0.8, 12.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Comparison of splitting methods as T varies:
      // Random row split leakage rate vs clean group entity split (0% leakage)
      const periods = [2, 3, 4, 6, 8, 10, 12]
      const pTest = 0.2
      const pTrain = 0.8
      // Probability of entity leaking across train and test: 1 - (pTrain^t + pTest^t)
      const leakageRate = periods.map((t) => {
        const unshared = Math.pow(pTrain, t) + Math.pow(pTest, t)
        return (1 - unshared) * 100
      })
      const cleanGroupSplit = periods.map(() => 0)

      const currLeakage = (1 - (Math.pow(pTrain, timePeriods) + Math.pow(pTest, timePeriods))) * 100

      return {
        x: periods,
        y: leakageRate,
        baseline: cleanGroupSplit,
        stat1: `Observation periods T = ${timePeriods}`,
        stat2: `Row-split entity leakage: ${currLeakage.toFixed(1)}%`,
        stat3: 'Group split: 0% leakage',
        warning:
          currLeakage > 40
            ? 'Catastrophic autocorrelation leakage: Random row splitting mixes entity history across train and test partitions'
            : 'Group-entity or temporal cutoff splits must be enforced to evaluate true generalization',
        axisXLabel: 'Repeated Periods per Entity (T)',
        axisYLabel: 'Cross-Fold Entity Overlap Contamination Rate (%)',
        xRange: [1.5, 12.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [visualizerMode, entityCount, timePeriods])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Panel & Longitudinal Data: Variance Decomposition, Cohort Attrition, and Split Leakage"
      state={state}
      hoverReadout={false}
      caption="Structural properties of longitudinal panels: Intraclass variance decomposition between entities and time periods, cumulative attrition creating unbalanced panels, and evaluation leakage under naive row-splitting."
      readouts={
        <>
          <Readout label="Panel Scale" value={results.stat1} />
          <Readout label="Longitudinal Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'within_vs_between'
                ? 'Within-Unit Var'
                : visualizerMode === 'attrition_dynamics'
                  ? 'Active Units'
                  : 'Split Integrity'
            }
            value={results.stat3}
            color={visualizerMode === 'splitting_paradigms' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Methodological Caution" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Panel Data Structural Trajectory">
        <Curve
          name={
            visualizerMode === 'within_vs_between'
              ? 'Within-Entity Variance Share (%)'
              : visualizerMode === 'attrition_dynamics'
                ? 'Balanced Panel Ideal (100% Retention)'
                : 'Group-Entity Partition Benchmark (0% Leakage)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'within_vs_between'
              ? 'Between-Entity Variance Share (%)'
              : visualizerMode === 'attrition_dynamics'
                ? 'Observed Unbalanced Cohort Retention'
                : 'Naive Row-Split Cross-Contamination Rate'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Coordinate" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

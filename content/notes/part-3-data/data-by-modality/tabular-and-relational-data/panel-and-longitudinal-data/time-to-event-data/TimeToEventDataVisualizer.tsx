import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TimeToEventDataVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'kaplan_meier_censoring' | 'landmarking_dynamics' | 'competing_risks'>(
      [
        {
          value: 'kaplan_meier_censoring',
          label: 'Right Censoring: Kaplan-Meier Risk Set vs Naive Imputation (Kaplan 1958)',
        },
        {
          value: 'landmarking_dynamics',
          label: 'Landmarking: Dynamic Risk Prediction & Landmark Windows (van Houwelingen 2007)',
        },
        { value: 'competing_risks', label: 'Competing Risks: Cumulative Incidence vs Kaplan-Meier Bias' },
      ],
      'kaplan_meier_censoring',
      { label: 'survival analysis perspective' },
    ),
    cohortSize: slider(100, 1000, 400, {
      step: 50,
      label: 'initial study cohort N',
    }),
    censoringRate: slider(10, 70, 35, {
      step: 5,
      label: 'right-censoring proportion (% censored)',
    }),
  })

  const { visualizerMode, cohortSize, censoringRate } = state

  const results = useMemo(() => {
    if (visualizerMode === 'kaplan_meier_censoring') {
      // Time grid from 0 to 60 months
      const times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
      // True survival probability follows exponential / Weibull decay: S(t) = exp(-0.02 * t)
      const trueSurvival = times.map((t) => Math.exp(-0.022 * t) * 100)
      // Naive imputation: treating censored time as failure time severely underestimates survival: S_naive(t) = exp(-(0.022 + 0.0006 * C) * t)
      const naivePenalty = 0.00035 * censoringRate
      const naiveSurvival = times.map((t) => Math.exp(-(0.022 + naivePenalty) * t) * 100)

      const currSurvival60 = Math.exp(-0.022 * 60) * 100
      const currNaive60 = Math.exp(-(0.022 + naivePenalty) * 60) * 100
      const biasGap = currSurvival60 - currNaive60

      return {
        x: times,
        y: trueSurvival,
        baseline: naiveSurvival,
        stat1: `Cohort N = ${cohortSize}`,
        stat2: `Censored rate: ${censoringRate}%`,
        stat3: `60-mo true vs naive: ${currSurvival60.toFixed(0)}% vs ${currNaive60.toFixed(0)}%`,
        warning:
          biasGap > 10
            ? 'Severe censoring bias: Treating right-censored durations as exact failure times severely underestimates patient survival'
            : 'Kaplan-Meier hazard estimation properly accounts for censored units dropping out of active risk sets',
        axisXLabel: 'Follow-Up Duration (Months t)',
        axisYLabel: 'Estimated Survival Probability S(t) (%)',
        xRange: [-2, 62] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'landmarking_dynamics') {
      // Landmark times s from 0 to 36 months, predicting survival across horizon s + 24
      const landmarks = [0, 6, 12, 18, 24, 30, 36]
      // Risk cohort size surviving at landmark s: N * exp(-0.02 * s)
      const survivingFraction = landmarks.map((s) => Math.round(Math.exp(-0.02 * s) * 100))
      // Fixed baseline: 100% initial cohort
      const initialBaseline = landmarks.map(() => 100)

      const currS = 18
      const currRiskFraction = Math.round(Math.exp(-0.02 * currS) * 100)
      const currActiveN = Math.round((cohortSize * currRiskFraction) / 100)

      return {
        x: landmarks,
        y: survivingFraction,
        baseline: initialBaseline,
        stat1: `Prediction window: +24 months`,
        stat2: `At landmark s=18m: ${currRiskFraction}% alive`,
        stat3: `Active risk set: ${currActiveN} units`,
        warning:
          'Landmarking resets the study baseline to time s, updating time-varying features strictly up to s and discarding prior events',
        axisXLabel: 'Landmark Eligibility Timestamp s (Months)',
        axisYLabel: 'Eligible Event-Free Risk Cohort Fraction (%)',
        xRange: [-2, 38] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Competing risks: Cause 1 (Primary event) vs Cause 2 (Competing event)
      const times = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
      // Cumulative incidence function CIF_1(t): true probability of Cause 1 occurring first
      const cif1 = times.map((t) => (1 - Math.exp(-0.015 * t)) * 0.65 * 100)
      // Standard 1 - KM(t) treating Cause 2 as censored: overestimates incidence!
      const kmBiasedIncidence = times.map((t) => (1 - Math.exp(-0.025 * t)) * 100)

      const currCIF = (1 - Math.exp(-0.015 * 60)) * 0.65 * 100
      const currKMOverestimate = (1 - Math.exp(-0.025 * 60)) * 100

      return {
        x: times,
        y: cif1,
        baseline: kmBiasedIncidence,
        stat1: `Cohort N = ${cohortSize}`,
        stat2: `60-mo true CIF: ${currCIF.toFixed(1)}%`,
        stat3: `Biased KM: ${currKMOverestimate.toFixed(1)}%`,
        warning:
          'CRITICAL ERROR: Treating competing risks as independent censoring violates non-informative censoring, artificially inflating cumulative incidence',
        axisXLabel: 'Follow-Up Duration (Months t)',
        axisYLabel: 'Cumulative Event Incidence (%)',
        xRange: [-2, 62] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [visualizerMode, cohortSize, censoringRate])

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
      title="Time-to-Event Data: Right-Censoring Mechanics, Landmarking, and Competing Risks"
      state={state}
      hoverReadout={false}
      caption="Evaluating survival and duration modeling: Risk set adjustment under right censoring, dynamic prediction through landmark cohort conditioning, and cumulative incidence bias under competing risks."
      readouts={
        <>
          <Readout label="Study Cohort" value={results.stat1} />
          <Readout label="Censoring Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'kaplan_meier_censoring'
                ? '60-Mo Survival'
                : visualizerMode === 'landmarking_dynamics'
                  ? 'Active Units'
                  : 'Primary Incidence'
            }
            value={results.stat3}
            color={visualizerMode === 'competing_risks' ? 'var(--red-600)' : 'var(--blue-600)'}
          />
          <Readout label="Methodological Directive" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Survival & Risk Trajectory">
        <Curve
          name={
            visualizerMode === 'kaplan_meier_censoring'
              ? 'Naive Imputation (Censoring Treated as Event)'
              : visualizerMode === 'landmarking_dynamics'
                ? 'Initial Baseline Cohort (100%)'
                : 'Biased 1 - KM Treating Competing Risks as Censored'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'kaplan_meier_censoring'
              ? 'True Kaplan-Meier Survival S(t)'
              : visualizerMode === 'landmarking_dynamics'
                ? 'Eligible Event-Free Risk Cohort at Landmark s'
                : 'True Cumulative Incidence Function CIF_1(t)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Measurement" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

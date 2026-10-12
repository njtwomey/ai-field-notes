import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CensoringCoarseningVisualizer() {
  const state = useFigureState({
    censoringThreshold: slider(15, 50, 30, {
      step: 1,
      label: 'study observation window cutoff C (days)',
    }),
    viewMode: choice<'survival_km' | 'scatter_censoring'>(
      [
        { value: 'survival_km', label: 'Kaplan-Meier Survival Probability S(t) vs Naive Empirical CDF' },
        { value: 'scatter_censoring', label: 'Censored Observations & Tobit Threshold Boundary' },
      ],
      'survival_km',
      { label: 'censoring analysis' },
    ),
  })

  const results = useMemo(() => {
    // 30 deterministic survival durations (days)
    const trueTimes = [
      8, 11, 14, 16, 19, 21, 23, 25, 27, 28, 31, 33, 35, 36, 38, 40, 42, 44, 46, 48, 50, 52, 55, 58, 60, 63, 66, 70, 75,
      80,
    ]
    const C = state.censoringThreshold

    let numCensored = 0
    const observedTimes: number[] = []
    const eventFlags: number[] = [] // 1 = observed event, 0 = censored at C

    trueTimes.forEach((t) => {
      if (t <= C) {
        observedTimes.push(t)
        eventFlags.push(1)
      } else {
        observedTimes.push(C)
        eventFlags.push(0)
        numCensored += 1
      }
    })

    const censoringPct = (numCensored / trueTimes.length) * 100.0
    const trueMean = trueTimes.reduce((a, b) => a + b, 0) / trueTimes.length
    const naiveMean = observedTimes.reduce((a, b) => a + b, 0) / observedTimes.length

    // Kaplan-Meier curve calculation across grid t in [0..60]
    const tGrid: number[] = []
    const kmCurve: number[] = []
    const naiveCurve: number[] = []

    // Distinct event times sorted
    const sortedObs = observedTimes.map((t, idx) => ({ t, event: eventFlags[idx] })).sort((a, b) => a.t - b.t)

    for (let curT = 0; curT <= 60; curT += 2) {
      tGrid.push(curT)

      // Naive survival: fraction of all 30 points that have observed time > curT
      const naiveSurv = sortedObs.filter((o) => o.t > curT).length / sortedObs.length
      naiveCurve.push(naiveSurv)

      // KM survival: product_{t_i <= curT, event=1} (1 - d_i / n_i)
      let s = 1.0
      let atRisk = sortedObs.length
      const distinctTimes = Array.from(new Set(sortedObs.map((o) => o.t))).sort((a, b) => a - b)

      distinctTimes.forEach((dt) => {
        if (dt <= curT) {
          const eventsAtDt = sortedObs.filter((o) => o.t === dt && o.event === 1).length
          const totalAtDt = sortedObs.filter((o) => o.t === dt).length
          if (atRisk > 0 && eventsAtDt > 0) {
            s *= 1.0 - eventsAtDt / atRisk
          }
          atRisk -= totalAtDt
        }
      })
      kmCurve.push(curT <= C ? s : (kmCurve[kmCurve.length - 1] ?? s))
    }

    // For scatter plot: x = index 1..30, y = observedTime
    const xIndices = trueTimes.map((_, idx) => idx + 1)
    const cutoffLine = xIndices.map(() => C)

    return {
      tGrid,
      kmCurve,
      naiveCurve,
      xIndices,
      observedTimes,
      cutoffLine,
      trueMean,
      naiveMean,
      censoringPct,
      numCensored,
    }
  }, [state.censoringThreshold])

  const axisX = useAxis({
    label: state.viewMode === 'survival_km' ? 'Duration / Timeline t (Days)' : 'Entity Observation Index',
    range: state.viewMode === 'survival_km' ? [0, 62] : [0.5, 30.5],
  })

  const axisY = useAxis({
    label: state.viewMode === 'survival_km' ? 'Survival Probability S(t)' : 'Recorded Duration (Days)',
    range: state.viewMode === 'survival_km' ? [0, 1.05] : [0, 60],
  })

  return (
    <Figure
      title="Right-Censoring and Information Loss in Duration Telemetry"
      state={state}
      hoverReadout={false}
      caption="When an observation window terminates at time C, entities that have not yet failed or churned are right-censored. Treating censored times as actual event occurrences produces severe downward bias in duration estimates; Kaplan-Meier and Tobit models properly condition on survival."
      readouts={
        <>
          <Readout label="True Population Mean" value={`${results.trueMean.toFixed(1)} days`} />
          <Readout
            label="Naive Sample Mean"
            value={`${results.naiveMean.toFixed(1)} days (-${(results.trueMean - results.naiveMean).toFixed(1)}d downward bias)`}
          />
          <Readout
            label="Right-Censored Volume"
            value={`${results.numCensored} of 30 units (${results.censoringPct.toFixed(0)}%)`}
          />
          <Readout
            label="Estimator Recommendation"
            value={results.censoringPct > 20 ? 'Mandatory Tobit / Kaplan-Meier' : 'Standard OLS plausible'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Censored Survival Profile vs Naive Complete-Case Distortion">
        {state.viewMode === 'survival_km' ? (
          <>
            <Curve
              name="Kaplan-Meier S(t) (Unbiased Censored Product)"
              x={results.tGrid}
              y={results.kmCurve}
              slot={0}
              width={2.5}
            />
            <Curve
              name="Naive Empirical Retention"
              x={results.tGrid}
              y={results.naiveCurve}
              slot={2}
              dashed
              width={2}
            />
          </>
        ) : (
          <>
            <Points
              name="Observed Recorded Durations"
              x={results.xIndices}
              y={results.observedTimes}
              slot={0}
              size={7}
            />
            <Curve
              name="Observation Window Cutoff C"
              x={results.xIndices}
              y={results.cutoffLine}
              slot={2}
              dashed
              width={2}
            />
          </>
        )}
      </Plot>
    </Figure>
  )
}

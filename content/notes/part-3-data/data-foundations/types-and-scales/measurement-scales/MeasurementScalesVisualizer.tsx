import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MeasurementScalesVisualizer() {
  const state = useFigureState({
    scaleMode: choice<'ordinal_monotonic' | 'circular_periodic'>(
      [
        {
          value: 'ordinal_monotonic',
          label: 'Ordinal Invariance (Monotonic Transform preserves Ranks, distorts Mean)',
        },
        {
          value: 'circular_periodic',
          label: 'Circular Scale (Hour of Day: Linear Discontinuity vs Cyclical Embedding)',
        },
      ],
      'ordinal_monotonic',
      { label: 'measurement scale demonstration' },
    ),
    parameter: slider(0.2, 4.0, 1.0, {
      step: 0.1,
      label: 'distortion exponent (k in x^k) / target hour of day (h)',
    }),
  })

  const results = useMemo(() => {
    if (state.scaleMode === 'ordinal_monotonic') {
      // 5 ordered ranks [1, 2, 3, 4, 5]
      const ranks = [1, 2, 3, 4, 5]
      const k = state.parameter
      const transformed = ranks.map((r) => Math.pow(r, k))

      // Median rank is always rank 3 (value = 3^k)
      // Arithmetic mean varies dramatically
      const sumTrans = transformed.reduce((a, b) => a + b, 0)
      const meanVal = sumTrans / transformed.length
      const medianVal = transformed[2]

      // Rank order correlation (Spearman) is identically 1.0
      // Linear correlation is < 1.0 when k != 1

      return {
        xValues: ranks,
        yValues: transformed,
        meanVal,
        medianVal,
        kVal: k,
        statusText: `Ranks strictly preserved; arithmetic mean shifted to ${meanVal.toFixed(2)}`,
      }
    } else {
      // Hours of day 0..23
      const hours: number[] = []
      const linearDist: number[] = []
      const circularDist: number[] = []
      const refHour = Math.min(23, Math.round(state.parameter * 5.75)) // scale slider to 0..23

      for (let h = 0; h < 24; h += 1) {
        hours.push(h)
        // Linear distance: |h - refHour|
        linearDist.push(Math.abs(h - refHour))
        // Circular distance on 24h dial: min(|h - refHour|, 24 - |h - refHour|)
        const rawDiff = Math.abs(h - refHour)
        circularDist.push(Math.min(rawDiff, 24 - rawDiff))
      }

      const diffAtMidnight = Math.abs(23 - 0) // linear = 23, circular = 1

      return {
        xValues: hours,
        yValues: state.parameter > 2 ? circularDist : linearDist,
        hours,
        linearDist,
        circularDist,
        refHour,
        diffAtMidnight,
        meanVal: 0,
        medianVal: 0,
        statusText: `Reference hour: ${refHour}:00. Midnight boundary step: linear = 23h, circular = 1h`,
      }
    }
  }, [state.parameter, state.scaleMode])

  const axisX = useAxis({
    label: state.scaleMode === 'ordinal_monotonic' ? 'Ordinal Rank Index (1 to 5)' : 'Hour of Day (0 to 23)',
    range: state.scaleMode === 'ordinal_monotonic' ? [0.8, 5.2] : [0, 23.5],
  })

  const axisY = useAxis({
    label:
      state.scaleMode === 'ordinal_monotonic' ? 'Transformed Score f(x) = x^k' : 'Distance to Reference Hour (hours)',
    range: state.scaleMode === 'ordinal_monotonic' ? [0, 100] : [0, 24],
  })

  return (
    <Figure
      title="Stevens' Measurement Scales and Admissible Mathematical Operations"
      state={state}
      hoverReadout={false}
      caption="Nominal and ordinal scales do not support arithmetic subtraction or Euclidean distances. Monotonic transformations leave median and rank order invariant but warp arithmetic means, while circular quantities require trigonometric projections to avoid 23-hour boundary discontinuities."
      readouts={
        state.scaleMode === 'ordinal_monotonic' ? (
          <>
            <Readout label="Rank Order" value="Perfect Invariance (Spearman rho = 1.0)" />
            <Readout label="Median Transformed" value={`${results.medianVal.toFixed(2)}`} />
            <Readout label="Mean Transformed" value={`${results.meanVal.toFixed(2)}`} />
            <Readout label="Admissible Operation" value="Median / Quantiles (Mean is undefined)" />
          </>
        ) : (
          <>
            <Readout label="Reference Hour" value={`${results.refHour}:00`} />
            <Readout label="23:00 to 00:00 Linear Delta" value="23 hours (Discontinuous!)" />
            <Readout label="23:00 to 00:00 Cyclical Delta" value="1 hour (Geodesic continuous)" />
            <Readout label="Recommended Encoding" value="[sin(2*pi*h/24), cos(2*pi*h/24)]" />
          </>
        )
      }
    >
      <Plot x={axisX} y={axisY} title="Relational Structure vs Numerical Artifacts">
        {state.scaleMode === 'ordinal_monotonic' ? (
          <>
            <Curve name="Transformed Values f(x)" x={results.xValues} y={results.yValues} slot={0} width={2.5} />
            <Points name="Discrete Ranks" x={results.xValues} y={results.yValues} slot={1} size={8} />
          </>
        ) : (
          <>
            <Curve name="Linear Naive Distance" x={results.hours!} y={results.linearDist!} slot={2} dashed width={2} />
            <Curve name="True Cyclical Distance" x={results.hours!} y={results.circularDist!} slot={0} width={2.5} />
          </>
        )}
      </Plot>
    </Figure>
  )
}

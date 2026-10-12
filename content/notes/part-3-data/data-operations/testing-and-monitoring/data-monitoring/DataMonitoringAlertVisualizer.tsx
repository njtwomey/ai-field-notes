import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataMonitoringAlertVisualizer() {
  const state = useFigureState({
    metricFocus: choice<'missing_rate' | 'prediction_distribution' | 'label_delay'>(
      [
        {
          value: 'missing_rate',
          label: 'Feature Null / Missing Rate (Ingestion drops & broken joins: Breck et al. 2017)',
        },
        {
          value: 'prediction_distribution',
          label: 'Prediction Score Distribution Drift (Population & covariate shift: Sculley et al. 2015)',
        },
        { value: 'label_delay', label: 'Ground-Truth Label Latency (Delayed feedback & attribution lag)' },
      ],
      'missing_rate',
      { label: 'production monitoring signal' },
    ),
    thresholdZ: slider(1.5, 4.5, 3.0, {
      step: 0.1,
      label: 'alert threshold sensitivity (Z-score deviation sigma)',
    }),
  })

  const results = useMemo(() => {
    // Generate trade-off curves across threshold Z in [1.5, 4.5]
    const zGrid: number[] = []
    const falseAlarmGrid: number[] = []
    const detectionDelayGrid: number[] = []

    for (let z = 1.5; z <= 4.5; z += 0.1) {
      zGrid.push(z)

      // False alarms per month follow gaussian tail ~ exp(-z^2 / 2)
      const fa = Math.max(0.1, 120.0 * Math.exp(-0.4 * z * z))
      falseAlarmGrid.push(fa)

      // Time to detect real corruption (hours): higher threshold requires more persistent evidence
      const delay = 0.5 + 2.2 * Math.pow(z, 1.8)
      detectionDelayGrid.push(delay)
    }

    const currentZ = state.thresholdZ
    const currentFa = Math.max(0.1, 120.0 * Math.exp(-0.4 * currentZ * currentZ))
    const currentDelay = 0.5 + 2.2 * Math.pow(currentZ, 1.8)

    let operationalHealth = ''
    if (currentFa > 15.0) {
      operationalHealth = 'High Alert Fatigue (On-call engineers will ignore pages)'
    } else if (currentFa < 1.0) {
      operationalHealth = 'Conservative (Low noise, but delayed incident detection)'
    } else {
      operationalHealth = 'Optimal Calibration (Balanced signal-to-noise ratio)'
    }

    return {
      zGrid,
      falseAlarmGrid,
      detectionDelayGrid,
      currentZ,
      currentFa,
      currentDelay,
      operationalHealth,
    }
  }, [state.thresholdZ])

  const axisX = useAxis({
    label: 'alert threshold sensitivity (Z-score standard deviations)',
    range: [1.5, 4.5],
  })
  const axisY = useAxis({
    label: 'false alarms per month (alert fatigue)',
    range: [0, 45],
  })

  return (
    <Figure
      title="Production data monitoring: alert thresholds, false alarm fatigue, and detection latency"
      state={state}
      hoverReadout={false}
      caption="Calibrating production data monitoring thresholds (Sculley et al. 2015, Breck et al. 2017). Setting sensitive alert thresholds (Z < 2.0 sigma) catches upstream pipeline ingestion drops and schema breakages rapidly, but floods on-call engineering teams with dozens of spurious alerts per month, inducing alert fatigue. Calibrating thresholds near Z = 3.0 standard deviations preserves high signal-to-noise ratios while bounding incident detection latency."
      readouts={
        <>
          <Readout label="false alarm rate" value={`${results.currentFa.toFixed(1)} alerts / month`} />
          <Readout label="mean time to detect (MTTD)" value={`${results.currentDelay.toFixed(1)} hours`} />
          <Readout label="monitoring posture" value={results.operationalHealth} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="False alarm rate vs alert threshold sensitivity">
        <Curve
          name="false alarm frequency (alerts / month)"
          x={results.zGrid}
          y={results.falseAlarmGrid}
          slot={0}
          width={2.5}
        />
        <Points name="current alert operating point" x={[results.currentZ]} y={[results.currentFa]} slot={0} size={6} />
      </Plot>
    </Figure>
  )
}

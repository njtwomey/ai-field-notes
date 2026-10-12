import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function IndustrialMaintenanceVisualizer() {
  const state = useFigureState({
    analysisMode: choice<'run_to_failure' | 'sensor_drift'>(
      [
        { value: 'run_to_failure', label: 'Run-to-Failure Degradation (NASA C-MAPSS benchmark: Saxena et al. 2008)' },
        {
          value: 'sensor_drift',
          label: 'Sensor Calibration Step Jump (Recalibration baseline shift vs mechanical wear)',
        },
      ],
      'run_to_failure',
      { label: 'prognostic data regime' },
    ),
    cycleCount: slider(10, 250, 160, {
      step: 10,
      label: 'operating lifetime cycles elapsed t',
    }),
  })

  const results = useMemo(() => {
    // Failure threshold occurs at cycle 220
    const failureCycle = 220
    const inflectionCycle = 110

    const tGrid: number[] = []
    const sensorGrid: number[] = []

    for (let t = 10; t <= 250; t += 10) {
      tGrid.push(t)
      let sensorVal = 0

      if (state.analysisMode === 'run_to_failure') {
        // Flat baseline early, followed by exponential damage accumulation after inflection
        if (t <= inflectionCycle) {
          sensorVal = 642.0 + 0.5 * Math.sin(t / 5)
        } else {
          const dt = t - inflectionCycle
          sensorVal = 642.0 + 0.0022 * dt * dt
        }
      } else {
        // Sensor replacement / recalibration jump at cycle 100
        sensorVal = t < 100 ? 642.0 : 648.5
      }

      sensorGrid.push(sensorVal)
    }

    const t = state.cycleCount
    let currentSensor = 0
    let trueRul = 0
    let healthStatus = ''

    if (state.analysisMode === 'run_to_failure') {
      if (t <= inflectionCycle) {
        currentSensor = 642.0 + 0.5 * Math.sin(t / 5)
      } else {
        const dt = t - inflectionCycle
        currentSensor = 642.0 + 0.0022 * dt * dt
      }
      trueRul = Math.max(0, failureCycle - t)
      if (trueRul === 0) {
        healthStatus = 'Terminal Failure (Component seized / broken)'
      } else if (trueRul < 30) {
        healthStatus = 'Critical Imminent Failure (Immediate maintenance shutdown required)'
      } else if (trueRul < 110) {
        healthStatus = 'Active Degradation (Accelerating mechanical wear)'
      } else {
        healthStatus = 'Healthy Baseline (Zero observable wear signal)'
      }
    } else {
      currentSensor = t < 100 ? 642.0 : 648.5
      trueRul = Math.max(0, failureCycle - t)
      healthStatus =
        t >= 100 ? 'Sensor Baseline Shift (Transducer replaced & zero-point recalibrated)' : 'Pre-Replacement Baseline'
    }

    return {
      tGrid,
      sensorGrid,
      currentT: t,
      currentSensor,
      trueRul,
      healthStatus,
    }
  }, [state.analysisMode, state.cycleCount])

  const axisX = useAxis({
    label: 'operating cycles / runtime hours',
    range: [10, 250],
  })
  const axisY = useAxis({
    label: 'sensor telemetry measurement (temperature T24 °R / vibration g)',
    range: [638, 672],
  })

  return (
    <Figure
      title="Industrial sensor and maintenance data: run-to-failure trajectories and sensor drift"
      state={state}
      hoverReadout={false}
      caption="Evaluating prognostic health management (PHM) and Remaining Useful Life (RUL) estimation (Saxena et al. 2008, Jardine et al. 2006). In run-to-failure records (e.g. NASA C-MAPSS turbofans), machinery operates normally until an inflection threshold, after which wear accelerates exponentially. Training algorithms clip target RUL (piecewise linear target) during healthy operation. Sensor replacements introduce discrete step-function baseline jumps that must be differentiated from mechanical wear."
      readouts={
        <>
          <Readout label="remaining useful life (RUL)" value={`${results.trueRul} cycles`} />
          <Readout label="telemetry measurement" value={`${results.currentSensor.toFixed(1)}`} />
          <Readout label="machinery health state" value={results.healthStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Telemetry degradation trajectory vs operating cycles">
        <Curve
          name="sensor telemetry trajectory"
          x={results.tGrid}
          y={results.sensorGrid}
          slot={results.trueRul < 30 ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current operating inspection point"
          x={[results.currentT]}
          y={[results.currentSensor]}
          slot={results.trueRul < 30 ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}

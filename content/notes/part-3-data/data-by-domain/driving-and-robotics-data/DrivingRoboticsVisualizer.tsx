import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DrivingRoboticsVisualizer() {
  const state = useFigureState({
    operationalChallenge: choice<'calibration' | 'rare_scenarios'>(
      [
        {
          value: 'calibration',
          label: 'Extrinsic Sensor Calibration (LiDAR-Camera spatial alignment: Caesar et al. 2020)',
        },
        {
          value: 'rare_scenarios',
          label: 'Long-Tail Critical Scenarios (Waymo Open Dataset active mining: Sun et al. 2020)',
        },
      ],
      'calibration',
      { label: 'robotics data challenge' },
    ),
    parameter: slider(0.0, 3.0, 0.8, {
      step: 0.1,
      label: 'extrinsic rotation error (degrees) / corner-case mining threshold',
    }),
  })

  const results = useMemo(() => {
    // Generate projection error or long-tail coverage curve across parameter in [0.0, 3.0]
    const xGrid: number[] = []
    const yGrid: number[] = []

    for (let x = 0.0; x <= 3.0; x += 0.1) {
      xGrid.push(x)
      let val = 0
      if (state.operationalChallenge === 'calibration') {
        // Pixel displacement at 50m distance: Delta pixels = f * tan(theta)
        // With focal length ~1200 px, 1 deg ≈ 21 pixels
        val = 1200.0 * Math.tan((x * Math.PI) / 180.0)
      } else {
        // Model failure rate on rare scenarios drops as mining threshold increases
        val = 45.0 * Math.exp(-1.1 * x) + 2.0
      }
      yGrid.push(val)
    }

    const currentX = state.parameter
    let metricReadout = ''
    let impactStatus = ''
    let technicalRemedy = ''

    if (state.operationalChallenge === 'calibration') {
      const pxOffset = 1200.0 * Math.tan((currentX * Math.PI) / 180.0)
      metricReadout = `${pxOffset.toFixed(1)} pixels at 50m`
      impactStatus =
        pxOffset > 25.0 ? 'Severe Cross-Modal Misalignment (Sensor Fusion Failure)' : 'Acceptable Projection'
      technicalRemedy = 'Online self-calibration & targetless target re-alignment'
    } else {
      const failureRate = 45.0 * Math.exp(-1.1 * currentX) + 2.0
      metricReadout = `${failureRate.toFixed(1)}% corner-case error rate`
      impactStatus = failureRate < 10.0 ? 'Robust Generalization Across Long-Tail Scenarios' : 'High Collision Risk'
      technicalRemedy = 'Closed-loop simulation generation & hard-negative fleet mining'
    }

    return {
      xGrid,
      yGrid,
      currentX,
      currentY:
        state.operationalChallenge === 'calibration'
          ? 1200.0 * Math.tan((currentX * Math.PI) / 180.0)
          : 45.0 * Math.exp(-1.1 * currentX) + 2.0,
      metricReadout,
      impactStatus,
      technicalRemedy,
    }
  }, [state.operationalChallenge, state.parameter])

  const axisX = useAxis({
    label:
      state.operationalChallenge === 'calibration'
        ? 'extrinsic angular rotation error (degrees)'
        : 'active corner-case mining intensity',
    range: [0, 3],
  })
  const axisY = useAxis({
    label:
      state.operationalChallenge === 'calibration'
        ? '2D-3D projection displacement (pixels)'
        : 'critical scenario failure rate (%)',
    range: [0, state.operationalChallenge === 'calibration' ? 70 : 50],
  })

  return (
    <Figure
      title="Driving and robotics data: sensor calibration drift and long-tail safety scenarios"
      state={state}
      hoverReadout={false}
      caption="Evaluating multimodal sensor dynamics in autonomous driving and robotics (Caesar et al. 2020 nuScenes, Sun et al. 2020 Waymo). Sub-degree extrinsic calibration drift between LiDAR and RGB camera sensors produces substantial 2D-3D bounding box projection displacements at 50m distance, degrading sensor fusion. Real-world robotics safety is dominated by the long-tail of rare corner cases (e.g. erratic obstacles, dusk glares), requiring active fleet-mining and closed-loop synthetic generation."
      readouts={
        <>
          <Readout label="performance metric" value={results.metricReadout} />
          <Readout label="system safety posture" value={results.impactStatus} />
          <Readout label="data engineering remedy" value={results.technicalRemedy} />
        </>
      }
    >
      <Plot
        x={axisX}
        y={axisY}
        title={
          state.operationalChallenge === 'calibration'
            ? 'Projection offset vs calibration angular drift'
            : 'Failure rate vs corner-case mining intensity'
        }
      >
        <Curve
          name="system response curve"
          x={results.xGrid}
          y={results.yGrid}
          slot={state.operationalChallenge === 'calibration' ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current operating point"
          x={[results.currentX]}
          y={[results.currentY]}
          slot={state.operationalChallenge === 'calibration' ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}

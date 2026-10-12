import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MotionRepresentationsVisualizer() {
  const state = useFigureState({
    motionMode: choice<'accuracy_latency' | 'flow_estimation' | 'camera_isolation'>(
      [
        {
          value: 'accuracy_latency',
          label: 'Representation Trade-off: Frame Diff vs Codec MV vs Dense RAFT Flow (Wu 2018)',
        },
        {
          value: 'flow_estimation',
          label: 'Optical Flow Fidelity: True Velocity vs Brightness Constancy Error (Horn 1981)',
        },
        {
          value: 'camera_isolation',
          label: 'Ego-Motion Compensation: Camera Pan vs Foreground Action Isolation',
        },
      ],
      'accuracy_latency',
      { label: 'motion representation' },
    ),
    param1: slider(1, 20, 8, {
      step: 1,
      label: 'motion velocity / camera speed / model complexity',
    }),
    param2: slider(0, 10, 3, {
      step: 1,
      label: 'illumination noise / ego-motion pan magnitude',
    }),
  })

  const { motionMode, param1, param2 } = state

  const results = useMemo(() => {
    if (motionMode === 'accuracy_latency') {
      // 4 representation regimes: 1=Frame Diff, 2=Codec MV (CoViAR), 3=TV-L1 Flow, 4=RAFT Flow
      const regimes = [1, 2, 3, 4]
      // Extraction latency in ms per frame (log scale visual)
      const latenciesMs = [0.15, 0.75, 38.0, 85.0]
      // Relative action recognition top-1 gain (%) on Kinetics/UCF101
      const accuracyGains = [2.1, 5.8, 8.4, 9.8]

      const currentIdx = Math.min(4, Math.max(1, Math.round((param1 / 20) * 3 + 1)))
      const currLatency = latenciesMs[currentIdx - 1]
      const currGain = accuracyGains[currentIdx - 1]
      const names = ['Frame Difference', 'Codec Motion Vectors (CoViAR)', 'TV-L1 Variational Flow', 'Deep RAFT Flow']
      const currName = names[currentIdx - 1]

      return {
        x: regimes,
        y: accuracyGains,
        baseline: latenciesMs.map((l) => Math.min(10, Math.log10(l + 1) * 4.8)), // scaled log latency
        stat1: `Method: ${currName}`,
        stat2: `Extraction Time: ${currLatency.toFixed(2)} ms/frame`,
        stat3: `Action Acc Gain: +${currGain.toFixed(1)}% Top-1`,
        warning:
          currentIdx >= 3
            ? 'Dense optical flow provides peak kinematic fidelity but requires days of offline GPU pre-extraction and massive terabyte storage arrays'
            : 'Codec motion vectors provide zero-cost in-stream motion cues from compressed P-frames, yielding 80% of flow benefits at 100x speed',
        axisXLabel: 'Representation Class (1=Diff, 2=Codec MVs, 3=TV-L1, 4=Deep RAFT)',
        axisYLabel: 'Action Recognition Accuracy Delta (%) / Log Compute Latency',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [0, 12] as [number, number],
      }
    } else if (motionMode === 'flow_estimation') {
      // True object displacement velocity from 1 to 20 pixels/frame
      const trueVelocities = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
      // Ideal linear estimation: u = v_true
      const idealFlow = trueVelocities.map((v) => v)
      // Estimated flow with brightness constancy violation and aperture damping
      const noise = param2 * 0.4
      const estimatedFlow = trueVelocities.map((v) => {
        // High velocities suffer sub-pixel matching attenuation without multi-scale pyramid
        const falloff = 1 / (1 + 0.02 * v)
        return Math.max(0, (v - noise * Math.sqrt(v)) * falloff)
      })

      const currV = param1
      const currEst = Math.max(0, (currV - noise * Math.sqrt(currV)) / (1 + 0.02 * currV))
      const flowError = Math.abs(currV - currEst)

      return {
        x: trueVelocities,
        y: estimatedFlow,
        baseline: idealFlow,
        stat1: `Ground Truth Speed: ${currV.toFixed(1)} px/frame`,
        stat2: `Estimated Flow: ${currEst.toFixed(1)} px/frame`,
        stat3: `Estimation Error: ${flowError.toFixed(2)} px/frame`,
        warning:
          param2 > 4
            ? 'Illumination flicker violates brightness constancy assumption dI/dt = 0; optical flow algorithm mistakes lighting changes for motion'
            : 'Multi-scale spatial pyramids successfully resolve large pixel displacements while preserving sharp velocity boundaries',
        axisXLabel: 'Physical Object True Velocity (Pixels per Frame)',
        axisYLabel: 'Estimated Optical Flow Velocity Magnitude (Pixels per Frame)',
        xRange: [0.5, 21] as [number, number],
        yRange: [0, 22] as [number, number],
      }
    } else {
      // Background / Foreground spatial points across frame: 1 to 10
      const pointIndices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const camPanSpeed = param2 * 1.5 // e.g. 0 to 15 px
      const fgSpeed = param1 * 0.8 // e.g. 1 to 16 px

      // Points 1-6 are background (moving purely due to camera pan)
      // Points 7-10 are foreground object (moving with camPan + fgSpeed)
      const rawFlow = pointIndices.map((p) => (p <= 6 ? camPanSpeed : camPanSpeed + fgSpeed))
      // Homography-compensated ego-motion subtraction: background flow removed
      const isolatedFlow = pointIndices.map((p) => (p <= 6 ? 0.3 : fgSpeed))

      const bgAvg = camPanSpeed
      const snr = fgSpeed / (camPanSpeed + 0.1)

      return {
        x: pointIndices,
        y: isolatedFlow,
        baseline: rawFlow,
        stat1: `Camera Pan Speed: ${bgAvg.toFixed(1)} px/frame`,
        stat2: `Foreground Action: ${fgSpeed.toFixed(1)} px/frame`,
        stat3: `Motion Signal-to-Noise: ${snr.toFixed(2)}x`,
        warning:
          camPanSpeed > fgSpeed
            ? 'Camera ego-motion dominates the optical flow field; without background homography stabilization, models learn camera shakes rather than user actions'
            : 'Compensated flow isolates intrinsic foreground body mechanics from camera pan and tilt',
        axisXLabel: 'Spatial Probe Location (1-6: Background Scene, 7-10: Target Subject)',
        axisYLabel: 'Motion Vector Magnitude (Pixels per Frame)',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, Math.max(...rawFlow) * 1.25 + 2] as [number, number],
      }
    }
  }, [motionMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Motion Representations: Optical Flow, Frame Differences, and Codec Vectors"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Representation Mode" value={results.stat1} />
          <Readout label="Throughput / Signal" value={results.stat2} />
          <Readout label="Performance Delta" value={results.stat3} />
          <Readout label="Kinematic Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Kinematic Vector Response and Resource Footprint">
        <Curve
          name={
            motionMode === 'accuracy_latency'
              ? 'Normalized Extraction Latency (Log Scale)'
              : motionMode === 'flow_estimation'
                ? 'Ideal Uncorrupted Velocity Truth'
                : 'Raw Uncompensated Optical Flow Field'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            motionMode === 'accuracy_latency'
              ? 'Action Recognition Top-1 Accuracy Gain (%)'
              : motionMode === 'flow_estimation'
                ? 'Estimated Flow Velocity Magnitude (px/frame)'
                : 'Compensated Foreground Action Motion Field'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Probe Measurements" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

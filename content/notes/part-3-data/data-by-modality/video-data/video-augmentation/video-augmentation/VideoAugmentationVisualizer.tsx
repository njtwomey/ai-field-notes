import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function VideoAugmentationVisualizer() {
  const state = useFigureState({
    augMode: choice<'temporal_consistency' | 'speed_scaling' | 'tube_masking'>(
      [
        {
          value: 'temporal_consistency',
          label: 'Spatial Consistency: Parameter Tying vs Uncoordinated Frame Flicker',
        },
        {
          value: 'speed_scaling',
          label: 'Temporal Speed: Playback Dilation alpha vs Kinematic Aliasing',
        },
        {
          value: 'tube_masking',
          label: 'VideoMAE Masking: Spatiotemporal Tube Masking Ratio rho (Tong 2022)',
        },
      ],
      'temporal_consistency',
      { label: 'augmentation strategy' },
    ),
    param1: slider(1, 20, 10, {
      step: 1,
      label: 'transform intensity / speed factor / mask ratio',
    }),
    param2: slider(4, 32, 16, {
      step: 4,
      label: 'temporal clip frames T / patch tube duration',
    }),
  })

  const { augMode, param1, param2 } = state

  const results = useMemo(() => {
    if (augMode === 'temporal_consistency') {
      // 16 frames in the clip: frame index 1 to 16
      const frames = Array.from({ length: param2 }, (_, i) => i + 1)
      const jitterIntensity = param1 * 0.5

      // Consistent spatial transform: identical rotation/crop across all frames
      // Smooth natural trajectory displacement
      const consistentDisplacement = frames.map((f) => 2.5 + Math.sin(f * 0.35) * 1.8)

      // Uncoordinated independent per-frame transform: high variance white-noise jumps
      const uncoordinatedDisplacement = frames.map((f) => {
        const base = 2.5 + Math.sin(f * 0.35) * 1.8
        const noise = Math.sin(f * 2.8) * jitterIntensity * 1.4
        return Math.max(0, base + Math.abs(noise))
      })

      const currJitter = (param1 * 0.5).toFixed(1)
      const avgNoiseSpike = (jitterIntensity * 1.4).toFixed(1)
      const isDangerous = param1 > 8

      return {
        x: frames,
        y: uncoordinatedDisplacement,
        baseline: consistentDisplacement,
        stat1: `Augmentation Strength: ${currJitter}x`,
        stat2: `Flicker Noise Magnitude: ±${avgNoiseSpike} px`,
        stat3: `Consistency: ${isDangerous ? 'STROBING / CORRUPT' : 'CONSISTENT / TIED'}`,
        warning: isDangerous
          ? 'Independent per-frame transforms destroy temporal optical flow continuity; high-frequency strobing injects artificial motion noise'
          : 'Tying spatial transformation parameters across all T frames preserves physical kinematic velocity fields and smooth motion trajectories',
        axisXLabel: 'Clip Frame Index (1 to T)',
        axisYLabel: 'Inter-Frame Spatial Displacement Delta (Pixels)',
        xRange: [0.5, param2 + 0.5] as [number, number],
        yRange: [0, Math.max(...uncoordinatedDisplacement) * 1.25 + 2] as [number, number],
      }
    } else if (augMode === 'speed_scaling') {
      // Speed multiplier alpha from 0.4x to 2.5x
      const speedFactors = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]
      // Information retention / Kinematic fidelity score (%)
      const fidelityScores = speedFactors.map((s) => {
        // optimal at 1.0x, drops under extreme slow (freezing) or extreme fast (aliasing)
        if (s <= 1.0) {
          return Math.max(40, 95 - Math.pow(1.0 - s, 1.4) * 55)
        } else {
          return Math.max(25, 95 - Math.pow((s - 1.0) / 1.5, 1.8) * 72)
        }
      })

      // Frame dropping rate (%) under speed acceleration
      const droppedFramesPct = speedFactors.map((s) => (s > 1.0 ? Math.min(65, (1.0 - 1.0 / s) * 100) : 0))

      const currSpeed = 0.4 + (param1 / 20) * 2.1
      const currFidelity =
        currSpeed <= 1.0
          ? Math.max(40, 95 - Math.pow(1.0 - currSpeed, 1.4) * 55)
          : Math.max(25, 95 - Math.pow((currSpeed - 1.0) / 1.5, 1.8) * 72)
      const currDrop = currSpeed > 1.0 ? Math.min(65, (1.0 - 1.0 / currSpeed) * 100) : 0

      return {
        x: speedFactors,
        y: fidelityScores,
        baseline: droppedFramesPct,
        stat1: `Speed Scaling: ${currSpeed.toFixed(2)}x playback`,
        stat2: `Kinematic Fidelity: ${currFidelity.toFixed(1)}%`,
        stat3: `Dropped Frame Rate: ${currDrop.toFixed(1)}%`,
        warning:
          currSpeed > 1.8
            ? 'High temporal acceleration s > 1.8x skips critical intermediate motion phases, causing ballistic action aliasing'
            : currSpeed < 0.6
              ? 'Excessive deceleration s < 0.6x produces redundant frozen duplicates, reducing the effective temporal receptive field'
              : 'Moderate speed jitter (0.8x - 1.3x) trains models invariant to diverse human action tempos',
        axisXLabel: 'Playback Speed Perturbation Multiplier alpha (x Speed)',
        axisYLabel: 'Kinematic Information Fidelity (%) / Dropped Frame Rate (%)',
        xRange: [0.35, 2.55] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Masking ratio rho from 0.1 to 0.95
      const maskRatios = [0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.85, 0.9, 0.95]
      // VideoMAE Self-Supervised downstream fine-tuning accuracy (%) on Kinetics-400
      // VideoMAE peaks at extremely high masking rates (90%) due to massive spatio-temporal redundancy!
      const fineTuneAcc = maskRatios.map((r) => {
        if (r < 0.85) {
          return 74 + Math.pow(r / 0.85, 1.2) * 8.5
        } else {
          return 82.5 - Math.pow((r - 0.85) / 0.1, 1.8) * 3.5
        }
      })

      // Image MAE baseline accuracy curve (peaks lower, around 75% masking, plunges past 85%)
      const imageMaeAcc = maskRatios.map((r) => {
        if (r < 0.75) {
          return 71 + Math.pow(r / 0.75, 1.2) * 6.5
        } else {
          return 77.5 - Math.pow((r - 0.75) / 0.2, 1.6) * 12.0
        }
      })

      const currRatio = 0.1 + (param1 / 20) * 0.85
      const currAcc =
        currRatio < 0.85
          ? 74 + Math.pow(currRatio / 0.85, 1.2) * 8.5
          : 82.5 - Math.pow((currRatio - 0.85) / 0.1, 1.8) * 3.5
      const computeSaved = (currRatio * 100).toFixed(0)

      return {
        x: maskRatios,
        y: fineTuneAcc,
        baseline: imageMaeAcc,
        stat1: `Tube Mask Ratio: rho = ${(currRatio * 100).toFixed(0)}%`,
        stat2: `Downstream Accuracy: ${currAcc.toFixed(1)}% Top-1`,
        stat3: `Compute FLOPs Saved: ~${computeSaved}%`,
        warning:
          currRatio >= 0.85
            ? 'Extremely high masking (85-90%) forces VideoMAE encoders to learn true temporal motion semantics rather than static spatial copying'
            : 'Low masking ratios (<70%) leave excessive spatio-temporal redundancy, allowing the model to trivially interpolate pixels from neighboring frames',
        axisXLabel: 'Spatiotemporal Masking Ratio rho (Fraction of Video Tokens Masked)',
        axisYLabel: 'Downstream Video Action Recognition Accuracy (%)',
        xRange: [0.08, 0.98] as [number, number],
        yRange: [60, 88] as [number, number],
      }
    }
  }, [augMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Video Augmentation: Temporal Parameter Consistency, Speed Scaling, and VideoMAE"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Transform Parameter" value={results.stat1} />
          <Readout label="Primary Trajectory Response" value={results.stat2} />
          <Readout label="Augmentation Health" value={results.stat3} />
          <Readout label="Methodological Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Video Augmentation Dynamics and Performance Response">
        <Curve
          name={
            augMode === 'temporal_consistency'
              ? 'Tied Consistent Spatial Transform (Ground Truth Motion)'
              : augMode === 'speed_scaling'
                ? 'Dropped Temporal Frame Rate (%)'
                : '2D Image MAE Baseline Transfer Curve (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            augMode === 'temporal_consistency'
              ? 'Uncoordinated Per-Frame Transform (Temporal Strobing)'
              : augMode === 'speed_scaling'
                ? 'Kinematic Information Fidelity (%)'
                : 'VideoMAE Spatiotemporal Tube Masking Transfer Curve (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Sample Configurations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

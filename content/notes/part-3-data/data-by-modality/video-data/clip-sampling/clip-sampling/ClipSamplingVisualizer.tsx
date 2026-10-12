import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ClipSamplingVisualizer() {
  const state = useFigureState({
    samplingMode: choice<'strategies_timeline' | 'stride_dynamics' | 'attention_complexity'>(
      [
        {
          value: 'strategies_timeline',
          label: 'Sampling Strategies: Dense Window vs Uniform vs TSN Segments (Wang 2016)',
        },
        {
          value: 'stride_dynamics',
          label: 'Temporal Stride: Nyquist Motion Aliasing vs Activity Horizon',
        },
        {
          value: 'attention_complexity',
          label: 'Video Transformer Complexity: Temporal Frames T vs Memory FLOPs (Arnab 2021)',
        },
      ],
      'strategies_timeline',
      { label: 'clip sampling regime' },
    ),
    sampleFrames: slider(4, 32, 16, {
      step: 4,
      label: 'number of sampled frames T',
    }),
    strideOrDuration: slider(1, 16, 4, {
      step: 1,
      label: 'temporal stride tau / action speed parameter',
    }),
  })

  const { samplingMode, sampleFrames, strideOrDuration } = state

  const results = useMemo(() => {
    if (samplingMode === 'strategies_timeline') {
      // 300 total video frames (~10 seconds at 30 FPS)
      const frameIndices = Array.from({ length: sampleFrames }, (_, i) => i + 1)
      const totalFrames = 300

      // Dense sampling: starts at frame 60, takes sampleFrames with stride tau
      const denseFrames = frameIndices.map((i) => Math.min(totalFrames, 60 + (i - 1) * strideOrDuration))
      // Uniform sampling: evenly spaced across entire 300 frames
      const uniformFrames = frameIndices.map((i) => Math.round(((i - 0.5) / sampleFrames) * totalFrames))

      const denseCoverageSec = ((sampleFrames * strideOrDuration) / 30).toFixed(1)
      const uniformIntervalSec = (totalFrames / sampleFrames / 30).toFixed(2)

      return {
        x: frameIndices,
        y: uniformFrames,
        baseline: denseFrames,
        stat1: `Sampled Frames: T = ${sampleFrames}`,
        stat2: `Dense Window: ${denseCoverageSec}s coverage (stride ${strideOrDuration})`,
        stat3: `Uniform Interval: ${uniformIntervalSec}s inter-frame gap`,
        warning:
          sampleFrames * strideOrDuration < 60
            ? 'Dense sampling captures contiguous micro-motion vectors but overlooks long-term action evolution across the full video'
            : 'Uniform sampling spans the full macro narrative but incurs temporal aliasing over rapid sub-second gestures',
        axisXLabel: 'Sampled Frame Sequence Index (1 to T)',
        axisYLabel: 'Raw Video Frame Timeline Position (0 to 300 frames)',
        xRange: [0.5, sampleFrames + 0.5] as [number, number],
        yRange: [0, 310] as [number, number],
      }
    } else if (samplingMode === 'stride_dynamics') {
      // Action speed from 0.5s (burst e.g. clapping) to 10.0s (gradual e.g. cooking)
      const actionSpeeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
      // Information capture efficiency (%) for chosen stride
      // Stride tau = strideOrDuration
      const infoScores = actionSpeeds.map((dur) => {
        // optimal stride is proportional to duration: tau_opt = dur * 30 / sampleFrames
        const tauOpt = Math.max(1, (dur * 30) / sampleFrames)
        const ratio = strideOrDuration / tauOpt
        // Bell-curve alignment around ratio = 1
        return Math.max(15, 95 * Math.exp(-0.7 * Math.pow(Math.log(ratio), 2)))
      })

      // Motion aliasing penalty if stride is too coarse for fast action
      const aliasingPenalty = actionSpeeds.map((dur) => {
        const minFramesForAction = (dur * 30) / strideOrDuration
        return minFramesForAction < 3 ? Math.min(85, (3 - minFramesForAction) * 35) : 5
      })

      const currDur = (strideOrDuration / 16) * 9.5 + 0.5
      const tauOpt = Math.max(1, (currDur * 30) / sampleFrames)
      const matchScore = Math.max(15, 95 * Math.exp(-0.7 * Math.pow(Math.log(strideOrDuration / tauOpt), 2)))

      return {
        x: actionSpeeds,
        y: infoScores,
        baseline: aliasingPenalty,
        stat1: `Current Stride: tau = ${strideOrDuration}`,
        stat2: `Action Horizon: ${currDur.toFixed(1)}s`,
        stat3: `Kinematic Match: ${matchScore.toFixed(0)}%`,
        warning:
          strideOrDuration > 8
            ? 'High stride tau > 8 misses brief atomic actions (e.g. eye blinks, tennis racquet strikes); temporal sampling falls below the Nyquist rate'
            : 'Low stride tau preserves micro-dynamics and optical flow coherence at the cost of narrower temporal context',
        axisXLabel: 'Physical Action Duration Horizon (Seconds)',
        axisYLabel: 'Temporal Sampling Quality / Aliasing Penalty (%)',
        xRange: [0.3, 10.5] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else {
      // Frame counts T from 4 to 32
      const frameCounts = [4, 8, 12, 16, 20, 24, 28, 32]
      // Full Spatio-Temporal Self-Attention FLOPs: proportional to (T * N_patch)^2
      // with N_patch = 196 (14x14 patches on 224x224)
      const jointAttentionFlops = frameCounts.map((t) => Math.pow(t * 196, 2) / 1e6)
      // Factorized Spatio-Temporal Attention (ViViT Model 2): T * (196^2) + 196 * (T^2)
      const factorizedFlops = frameCounts.map((t) => (t * Math.pow(196, 2) + 196 * Math.pow(t, 2)) / 1e6)

      const currT = sampleFrames
      const currJoint = (Math.pow(currT * 196, 2) / 1e6).toFixed(1)
      const currFactorized = ((currT * Math.pow(196, 2) + 196 * Math.pow(currT, 2)) / 1e6).toFixed(1)
      const speedup = (Number(currJoint) / Number(currFactorized)).toFixed(1)

      return {
        x: frameCounts,
        y: jointAttentionFlops,
        baseline: factorizedFlops,
        stat1: `Clip Frames: T = ${currT}`,
        stat2: `Joint Attention: ${currJoint} M-FLOPs`,
        stat3: `Factorized: ${currFactorized} M-FLOPs (${speedup}x cheaper)`,
        warning:
          currT >= 24
            ? 'Quadratic self-attention scaling O(T^2 x S^2) exhausts high-end GPU VRAM; factorized spatio-temporal attention or temporal pooling is mandatory'
            : 'Linear scaling maintained under factorized space-time attention layers; enables scaling to longer video temporal contexts',
        axisXLabel: 'Sampled Temporal Frames per Clip (T)',
        axisYLabel: 'Attention Matrix Operations (Mega-FLOPs per Layer)',
        xRange: [3, 33] as [number, number],
        yRange: [0, Math.max(...jointAttentionFlops) * 1.15] as [number, number],
      }
    }
  }, [samplingMode, sampleFrames, strideOrDuration])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Clip and Frame Sampling: Strategies, Strides, and Computational Scaling"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Sampling Configuration" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Secondary Metric" value={results.stat3} />
          <Readout label="Sampling Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Temporal Sampling Distribution and Complexity Profile">
        <Curve
          name={
            samplingMode === 'strategies_timeline'
              ? 'Dense Window Frame Position (Frame ID)'
              : samplingMode === 'stride_dynamics'
                ? 'Motion Aliasing / Information Loss Penalty (%)'
                : 'Factorized Spatio-Temporal Attention (ViViT) FLOPs'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            samplingMode === 'strategies_timeline'
              ? 'Uniform Linspace Frame Position (Frame ID)'
              : samplingMode === 'stride_dynamics'
                ? 'Kinematic Information Match Efficiency (%)'
                : 'Full Joint Spatio-Temporal Attention FLOPs'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Temporal Sample Positions" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

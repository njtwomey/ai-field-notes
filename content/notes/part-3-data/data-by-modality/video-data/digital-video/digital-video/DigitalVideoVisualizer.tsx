import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DigitalVideoVisualizer() {
  const state = useFigureState({
    videoMode: choice<'gop_seek' | 'codec_bitrate' | 'fps_throughput' | 'vfr_drift'>(
      [
        {
          value: 'gop_seek',
          label: 'GOP Structure: Random Seek Latency vs Keyframe Interval (Wiegand 2003)',
        },
        {
          value: 'codec_bitrate',
          label: 'Codec Efficiency: Bitrate vs VMAF Quality (H.264 vs HEVC vs AV1)',
        },
        {
          value: 'fps_throughput',
          label: 'Temporal Sampling: Frame Rate FPS vs Tensor Ingestion Throughput',
        },
        {
          value: 'vfr_drift',
          label: 'Variable Frame Rate (VFR): PTS Clock Drift vs Audio Desynchronization',
        },
      ],
      'gop_seek',
      { label: 'digital video perspective' },
    ),
    param1: slider(10, 250, 60, {
      step: 10,
      label: 'GOP length / target bitrate / sampling FPS',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'decoder threads / clip duration / resolution preset',
    }),
  })

  const { videoMode, param1, param2 } = state

  const results = useMemo(() => {
    if (videoMode === 'gop_seek') {
      // GOP sizes from 10 to 250 frames
      const gopSizes = [10, 25, 50, 75, 100, 125, 150, 200, 250]
      // Average frames decoded per random seek: N_GOP / 2
      // Decode latency in milliseconds at ~1.2 ms per P/B frame (CPU software decode)
      const decodeLatenciesMs = gopSizes.map((g) => (g / 2) * 1.25 + 4.0)
      // HW-accelerated decode (NVDEC) baseline
      const hwLatenciesMs = gopSizes.map((g) => (g / 2) * 0.18 + 1.5)

      const currGop = param1
      const currSwLatency = (currGop / 2) * 1.25 + 4.0
      const currHwLatency = (currGop / 2) * 0.18 + 1.5
      const avgFrames = Math.round(currGop / 2)

      return {
        x: gopSizes,
        y: decodeLatenciesMs,
        baseline: hwLatenciesMs,
        stat1: `GOP Length: ${currGop} frames (I-frame period)`,
        stat2: `Avg Seek Frames: ${avgFrames} frames`,
        stat3: `CPU Latency: ${currSwLatency.toFixed(1)} ms | GPU: ${currHwLatency.toFixed(1)} ms`,
        warning:
          currGop >= 120
            ? 'Long GOP yields stellar compression but stalls random seek loaders; workers spend >90% of cycle time sequentially decoding discarded delta frames'
            : 'Frequent keyframes allow sub-10ms random seeks suitable for high-throughput clip sampling, at a modest 15-25% storage bit-cost',
        axisXLabel: 'Group of Pictures (GOP) Keyframe Distance (Frames between I-Frames)',
        axisYLabel: 'Average Random Access Seek Latency (Milliseconds ms)',
        xRange: [5, 260] as [number, number],
        yRange: [0, Math.max(...decodeLatenciesMs) * 1.15] as [number, number],
      }
    } else if (videoMode === 'codec_bitrate') {
      // Bitrate in Mbps from 0.5 to 12.0 Mbps
      const bitrates = [0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0]
      // VMAF Quality scores (0..100) for H.264/AVC (baseline) vs HEVC/AV1 (modern)
      const h264Vmaf = bitrates.map((b) => Math.min(99, 35 + 58 * (1 - Math.exp(-0.35 * b))))
      const hevcVmaf = bitrates.map((b) => Math.min(99, 48 + 50 * (1 - Math.exp(-0.6 * b))))

      const currBitrate = (param1 / 250) * 11.5 + 0.5
      const currH264 = Math.min(99, 35 + 58 * (1 - Math.exp(-0.35 * currBitrate)))
      const currHevc = Math.min(99, 48 + 50 * (1 - Math.exp(-0.6 * currBitrate)))
      const bitrateSaving = ((currBitrate - currBitrate * 0.55) / currBitrate) * 100

      return {
        x: bitrates,
        y: hevcVmaf,
        baseline: h264Vmaf,
        stat1: `Bitrate: ${currBitrate.toFixed(1)} Mbps`,
        stat2: `HEVC/AV1 VMAF: ${currHevc.toFixed(1)}`,
        stat3: `H.264 VMAF: ${currH264.toFixed(1)} (Saving: ~${bitrateSaving.toFixed(0)}%)`,
        warning:
          currBitrate < 2.0
            ? 'Low-bitrate regime: H.264 exhibits severe macroblocking and motion blur artifacts; modern HEVC/AV1 preserves high-frequency edge gradients'
            : 'High-bitrate regime: perceptual quality saturates near VMAF 95; training architectures gain negligible accuracy beyond 4 Mbps 1080p',
        axisXLabel: 'Encoded Video Bitrate (Megabits per Second Mbps)',
        axisYLabel: 'Perceptual Visual Quality Metric (Netflix VMAF Score 0-100)',
        xRange: [0.2, 12.5] as [number, number],
        yRange: [30, 102] as [number, number],
      }
    } else if (videoMode === 'fps_throughput') {
      // Sampling frame rates: 1, 2, 4, 8, 12, 16, 24, 30, 60 FPS
      const fpsRates = [1, 2, 4, 8, 12, 16, 24, 30, 60]
      // Uncompressed 1080p tensor data rate: 1920*1080*3 bytes = 6.22 MB per frame
      const dataRateMb = fpsRates.map((f) => f * 6.22)
      // Resized 224x224 tensor data rate: 224*224*3 bytes = 0.15 MB per frame
      const modelRateMb = fpsRates.map((f) => f * 0.15)

      const currFps = Math.min(60, Math.max(1, Math.round((param1 / 250) * 59 + 1)))
      const currRawThroughput = currFps * 6.22
      const currBatchThroughput = currFps * 0.15 * param2

      return {
        x: fpsRates,
        y: dataRateMb,
        baseline: modelRateMb.map((m) => m * 20), // scaled for visualization
        stat1: `Capture Rate: ${currFps} FPS`,
        stat2: `Raw 1080p Stream: ${currRawThroughput.toFixed(1)} MB/s`,
        stat3: `Batch Ingestion (x${param2}): ${currBatchThroughput.toFixed(1)} MB/s`,
        warning:
          currFps > 30
            ? 'High frame rates flood PCIe memory buses with temporally redundant neighboring frames without conveying novel action dynamics'
            : 'Strided temporal subsampling (e.g. 4-8 FPS) captures macro action mechanics while reducing memory bandwidth by 85%',
        axisXLabel: 'Temporal Sampling Frequency (Frames per Second FPS)',
        axisYLabel: 'Raw Uncompressed Pixel Throughput (Megabytes per Second MB/s)',
        xRange: [0, 64] as [number, number],
        yRange: [0, Math.max(...dataRateMb) * 1.15] as [number, number],
      }
    } else {
      // Video timeline seconds from 0 to 60s
      const timelineSec = [0, 5, 10, 15, 20, 30, 40, 50, 60]
      // Assuming nominal 30 FPS, but jittery VFR camera drops frames under thermal throttle
      // Drift accumulates: e.g. 15 ms per second of video
      const vfrDriftMs = timelineSec.map((t) => t * 14.5)
      // CFR zero drift baseline
      const cfrZero = timelineSec.map(() => 0)

      const currTime = Math.min(60, Math.max(0, (param1 / 250) * 60))
      const currDrift = currTime * 14.5
      const droppedFrames = Math.round((currDrift / 1000) * 30)

      return {
        x: timelineSec,
        y: vfrDriftMs,
        baseline: cfrZero,
        stat1: `Clip Timestamp: ${currTime.toFixed(1)}s`,
        stat2: `PTS Clock Drift: +${currDrift.toFixed(0)} ms`,
        stat3: `Frame Jitter: ~${droppedFrames} frames offset`,
        warning:
          currDrift > 100
            ? 'Cumulative VFR clock drift exceeds phoneme duration (>80 ms); naive index-based frame slicing completely breaks audio-visual lip sync'
            : 'Minor clock offset; within perceptual tolerance, but strictly requires PTS-based timestamp interpolation before multimodal alignment',
        axisXLabel: 'Video Playback Elapsed Time (Seconds)',
        axisYLabel: 'Cumulative Presentation Timestamp (PTS) Drift vs Nominal CFR (ms)',
        xRange: [0, 62] as [number, number],
        yRange: [-10, Math.max(...vfrDriftMs) * 1.15] as [number, number],
      }
    }
  }, [videoMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Digital Video: Codecs, GOP Structures, Decoding Latencies, and Clocks"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Video Parameter" value={results.stat1} />
          <Readout label="Pipeline Impact" value={results.stat2} />
          <Readout label="Hardware Metric" value={results.stat3} />
          <Readout label="Architectural Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Video Decoding and Performance Function">
        <Curve
          name={
            videoMode === 'gop_seek'
              ? 'Hardware Accelerated NVDEC Latency (ms)'
              : videoMode === 'codec_bitrate'
                ? 'Legacy H.264/AVC Baseline VMAF'
                : videoMode === 'fps_throughput'
                  ? 'Downscaled 224x224 Ingestion Bandwidth (MB/s x 20)'
                  : 'Constant Frame Rate (CFR) Zero Drift Reference'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            videoMode === 'gop_seek'
              ? 'CPU Software Sequential Decode Latency (ms)'
              : videoMode === 'codec_bitrate'
                ? 'Modern HEVC/AV1 Quality Curve'
                : videoMode === 'fps_throughput'
                  ? 'Uncompressed 1080p Pixel Stream Throughput (MB/s)'
                  : 'VFR Cumulative Presentation Timestamp Drift (ms)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Measured Benchmarks" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

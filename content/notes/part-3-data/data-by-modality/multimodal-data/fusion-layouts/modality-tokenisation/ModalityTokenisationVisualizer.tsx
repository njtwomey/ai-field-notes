import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ModalityTokenisationVisualizer() {
  const state = useFigureState({
    mode: choice<'token_rate_imbalance' | 'rvq_codebook_levels' | 'patch_size_scaling'>(
      [
        {
          value: 'token_rate_imbalance',
          label: 'Multimodal Sequence Imbalance: Tokens per Second Across Modalities (Dosovitskiy 2021)',
        },
        {
          value: 'rvq_codebook_levels',
          label: 'Neural Audio Codec (EnCodec): Residual VQ Stages M vs SNR & Bitrate (Défossez 2022)',
        },
        {
          value: 'patch_size_scaling',
          label: 'Patch Tokenisation (ViT / PatchTST): Patch Size P vs Sequence Length N (Nie 2023)',
        },
      ],
      'token_rate_imbalance',
      { label: 'multimodal tokenisation' },
    ),
    param1: slider(1, 10, 4, {
      step: 1,
      label: 'sequence duration T (s) / RVQ stages M / image resolution scale (x224)',
    }),
    param2: slider(8, 32, 16, {
      step: 4,
      label: 'patch size P (pixels) / audio framerate / quantization bits',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'token_rate_imbalance') {
      const durationS = param1 // 1 to 10 seconds
      const durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const textTokens: number[] = []
      const audioTokens: number[] = []
      const videoTokens: number[] = []

      for (const t of durations) {
        // Text: ~3.5 words/s * 1.3 tokens/word = ~4.5 tokens/s
        textTokens.push(Math.round(4.5 * t))
        // Audio EnCodec (24 kHz, 75 fps x 4 codebooks = 300 tokens/s)
        audioTokens.push(300 * t)
        // Video (15 fps, 196 patches/frame = 2,940 tokens/s)
        videoTokens.push(2940 * t)
      }

      const activeText = Math.round(4.5 * durationS)
      const activeAudio = 300 * durationS
      const activeVideo = 2940 * durationS
      const videoToTextRatio = activeVideo / Math.max(1, activeText)

      return {
        x: durations,
        y1: audioTokens,
        y2: videoTokens,
        stat1: `T = ${durationS}s duration`,
        stat2: `Text: ${activeText} tokens`,
        stat3: `Audio: ${activeAudio.toLocaleString()} tokens`,
        insight: `Video dominates sequence length (${activeVideo.toLocaleString()} tokens, ${videoToTextRatio.toFixed(0)}x text)`,
        axisXLabel: 'Physical Sequence Duration T (seconds)',
        axisYLabel: 'Generated Token Count N',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [0, 31000] as [number, number],
      }
    } else if (mode === 'rvq_codebook_levels') {
      const stagesM = param1 // 1 to 8 stages
      const stages = [1, 2, 3, 4, 5, 6, 7, 8]
      const bitratesKbps: number[] = []
      const snrReconstructionDB: number[] = []

      for (const m of stages) {
        // 75 Hz framerate * m codebooks * 10 bits / 1000 = 0.75 * m kbps
        const kbps = 75 * m * 0.01 // e.g. 1.5 kbps to 6.0 kbps
        // Each RVQ codebook level adds roughly ~4.5 dB SNR
        const snr = 12.0 + 4.2 * (m - 1)
        bitratesKbps.push(kbps)
        snrReconstructionDB.push(snr)
      }

      const activeKbps = 75 * stagesM * 0.01
      const activeSnr = 12.0 + 4.2 * (stagesM - 1)

      return {
        x: stages,
        y1: snrReconstructionDB,
        y2: bitratesKbps.map((b) => b * 6),
        stat1: `RVQ Depth M = ${stagesM} levels`,
        stat2: `Bitrate: ${activeKbps.toFixed(2)} kbps`,
        stat3: `Reconstruction SNR: ${activeSnr.toFixed(1)} dB`,
        insight: 'Residual VQ quantizes successive error residuals, trading bitrate linearly for exponential precision',
        axisXLabel: 'Residual VQ Codebook Depth M',
        axisYLabel: 'Reconstruction Quality SNR (dB) & Scaled Bitrate',
        xRange: [0.5, 8.5] as [number, number],
        yRange: [0, 50] as [number, number],
      }
    } else {
      // Patch size P vs token count N for 224x224 and 448x448 image
      const patchSizes = [8, 12, 14, 16, 20, 24, 28, 32]
      const tokens224: number[] = []
      const tokens448: number[] = []

      for (const p of patchSizes) {
        tokens224.push(Math.round(Math.pow(224 / p, 2)))
        tokens448.push(Math.round(Math.pow(448 / p, 2)))
      }

      const activeP = param2 // 8 to 32
      const activeN224 = Math.round(Math.pow(224 / activeP, 2))
      const activeN448 = Math.round(Math.pow(448 / activeP, 2))

      return {
        x: patchSizes,
        y1: tokens224,
        y2: tokens448,
        stat1: `Patch Size P = ${activeP}×${activeP} px`,
        stat2: `224px Image: ${activeN224} tokens`,
        stat3: `448px Image: ${activeN448} tokens`,
        insight: `Token sequence length scales as O(1/P²), while self-attention cost scales as O(N²)`,
        axisXLabel: 'Patch Dimension P (pixels)',
        axisYLabel: 'Token Sequence Length N = (H·W)/P²',
        xRange: [6, 34] as [number, number],
        yRange: [0, 3200] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Modality Tokenisation: Token Rates, Residual VQ Codebooks, and Patch Partitioning"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Tokenisation Config" value={results.stat1} />
          <Readout label="Primary Rate / Scale" value={results.stat2} />
          <Readout label="Target Dimension / Quality" value={results.stat3} />
          <Readout label="Multimodal Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Sequence Length and Compression Trade-Offs Across Modalities">
        <Curve
          name={
            mode === 'token_rate_imbalance'
              ? 'Audio Codec EnCodec Token Stream (tokens)'
              : mode === 'rvq_codebook_levels'
                ? 'Reconstruction Perceptual SNR (dB)'
                : 'Standard Resolution (224×224) Patch Token Count'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'token_rate_imbalance'
              ? 'Raw Video 15fps ViT Patch Stream (tokens)'
              : mode === 'rvq_codebook_levels'
                ? 'Transmission Bitrate Metric (Scaled kbps)'
                : 'High Resolution (448×448) Patch Token Count'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

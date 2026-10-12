import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AudioAugmentationVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'snr_mixing' | 'specaugment' | 'speed_perturbation'>(
      [
        { value: 'snr_mixing', label: 'Additive Noise: SNR Scaling vs Intelligibility & Noise Power (Ko 2015)' },
        { value: 'specaugment', label: 'SpecAugment: Frequency & Time Masking vs Information Retention (Park 2019)' },
        { value: 'speed_perturbation', label: 'Speed Perturbation: Tempo Rescaling vs Formant Shift Dynamics' },
      ],
      'snr_mixing',
      { label: 'augmentation strategy' },
    ),
    snrTargetDb: slider(-10, 25, 10, {
      step: 5,
      label: 'target signal-to-noise ratio SNR (dB)',
    }),
    maskSize: slider(5, 40, 20, {
      step: 5,
      label: 'SpecAugment mask width (bins / frames)',
    }),
  })

  const { visualizerMode, snrTargetDb, maskSize } = state

  const results = useMemo(() => {
    if (visualizerMode === 'snr_mixing') {
      // SNR sweep from -10 dB to 30 dB
      const snrs = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
      // Relative noise-to-signal power ratio: 10^(-SNR / 10) * 100
      const noisePowerShare = snrs.map((s) => {
        const pNoise = Math.pow(10, -s / 10)
        return (pNoise / (1 + pNoise)) * 100
      })
      // Clean speech power share: 100 - noisePowerShare
      const speechPowerShare = noisePowerShare.map((np) => 100 - np)

      const currNoiseRatio = Math.pow(10, -snrTargetDb / 10)
      const currSpeechShare = (1 / (1 + currNoiseRatio)) * 100

      return {
        x: snrs,
        y: speechPowerShare,
        baseline: noisePowerShare,
        stat1: `Target SNR = ${snrTargetDb} dB`,
        stat2: `Speech energy: ${currSpeechShare.toFixed(1)}%`,
        stat3: `Noise energy: ${(100 - currSpeechShare).toFixed(1)}%`,
        warning:
          snrTargetDb < 0
            ? 'Extreme noise regime: Noise power exceeds speech energy; tests acoustic feature robustness under severe distortion'
            : snrTargetDb > 20
              ? 'Mild noise regime: Minimal perceptual disruption; models learn invariance to light background room hum'
              : 'Balanced industrial noise regime: Realistic simulation of cafeteria, traffic, or factory environments',
        axisXLabel: 'Signal-to-Noise Ratio SNR (dB)',
        axisYLabel: 'Relative Power Share (%)',
        xRange: [-12, 32] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (visualizerMode === 'specaugment') {
      // 80 Mel frequency channels (standard filterbank)
      // Vary mask parameter F from 5 to 40
      const maskWidths = [5, 10, 15, 20, 25, 30, 35, 40]
      const totalBins = 80
      // Information retention %: (1 - 2 * F / 80) * 100 (assuming 2 masks)
      const retentionPercent = maskWidths.map((w) => Math.max(10, (1 - (2 * w) / totalBins) * 100))
      const safeBaseline = maskWidths.map(() => 50)

      const currRetention = Math.max(10, (1 - (2 * maskSize) / totalBins) * 100)

      return {
        x: maskWidths,
        y: retentionPercent,
        baseline: safeBaseline,
        stat1: `Mask width F = ${maskSize} channels`,
        stat2: `Channels masked: ${maskSize * 2} / 80`,
        stat3: `Retained spectrum: ${currRetention.toFixed(1)}%`,
        warning:
          maskSize >= 30
            ? 'Heavy spectral masking: Zeroing over 60% of frequency channels forces model to exploit broad temporal context'
            : 'Moderate SpecAugment: Standard speech regularization preventing co-adaptation to narrow frequency bands',
        axisXLabel: 'SpecAugment Mask Width Parameter (Channels / Frames)',
        axisYLabel: 'Spectrogram Retention Ratio (%)',
        xRange: [3, 42] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Speed perturbation factor: 0.9, 0.95, 1.0, 1.05, 1.1
      const speedFactors = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]
      // Effective duration scaling: 100 / factor (%)
      const durationScalings = speedFactors.map((s) => (1 / s) * 100)
      // Formant frequency shift: s * 100 (%)
      const formantShifts = speedFactors.map((s) => s * 100)

      return {
        x: speedFactors.map((s) => Math.round(s * 100)),
        y: durationScalings,
        baseline: formantShifts,
        stat1: 'Standard perturbation: ±10% speed',
        stat2: 'Transcript text preserved',
        stat3: '3x Dataset Expansion',
        warning:
          'Resampling at 0.9x and 1.1x speed scales duration while naturally shifting formants, simulating diverse vocal tract lengths',
        axisXLabel: 'Speed Perturbation Factor (% Baseline)',
        axisYLabel: 'Normalized Duration / Pitch Shift (%)',
        xRange: [82, 118] as [number, number],
        yRange: [80, 125] as [number, number],
      }
    }
  }, [visualizerMode, snrTargetDb, maskSize])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Audio Augmentation: Additive Noise SNR, SpecAugment Masking, and Speed Perturbation"
      state={state}
      hoverReadout={false}
      caption="Evaluating audio data augmentation: Calibrated SNR additive noise mixing, time and frequency masking in SpecAugment, and speed perturbation duration vs formant scaling."
      readouts={
        <>
          <Readout label="Augmentation Mode" value={results.stat1} />
          <Readout label="Transformation Scale" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'snr_mixing'
                ? 'Speech Energy'
                : visualizerMode === 'specaugment'
                  ? 'Spectrogram Kept'
                  : 'Data Diversity'
            }
            value={results.stat3}
            color={visualizerMode === 'specaugment' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Training Directive" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Audio Augmentation Dynamics">
        <Curve
          name={
            visualizerMode === 'snr_mixing'
              ? 'Noise Energy Share (%)'
              : visualizerMode === 'specaugment'
                ? '50% Minimum Retention Threshold'
                : 'Formant Frequency Shift (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'snr_mixing'
              ? 'Clean Speech Energy Share (%)'
              : visualizerMode === 'specaugment'
                ? 'Spectrogram Information Retention Ratio (%)'
                : 'Utterance Duration Scaling (% Baseline)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Configuration Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

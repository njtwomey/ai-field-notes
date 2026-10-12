import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function AudioPreprocessingVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'preemphasis_tilt' | 'loudness_normalisation' | 'vad_tradeoff'>(
      [
        { value: 'preemphasis_tilt', label: 'Pre-Emphasis Filter: High-Frequency Formant Boost (Rabiner 2007)' },
        { value: 'loudness_normalisation', label: 'Loudness Normalisation: Peak vs Gated Perceptual LUFS' },
        { value: 'vad_tradeoff', label: 'Voice Activity Detection: Energy Threshold vs Speech Clipping Rate' },
      ],
      'preemphasis_tilt',
      { label: 'preprocessing transformation' },
    ),
    filterAlpha: slider(0.85, 0.99, 0.97, {
      step: 0.02,
      label: 'pre-emphasis coefficient alpha',
    }),
    energyThresholdDb: slider(-50, -15, -30, {
      step: 5,
      label: 'VAD energy threshold (dBFS)',
    }),
  })

  const { visualizerMode, filterAlpha, energyThresholdDb } = state

  const results = useMemo(() => {
    if (visualizerMode === 'preemphasis_tilt') {
      // Frequency axis from 100 Hz to 8000 Hz across 11 points (log-spaced or linear speech band)
      const freqs = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
      const fs = 16000
      // Pre-emphasis filter magnitude response: |H(omega)|^2 = 1 + alpha^2 - 2*alpha*cos(2*pi*f/fs)
      // in dB: 10 * log10(...)
      const boostDb = freqs.map((f) => {
        const omega = (2 * Math.PI * f) / fs
        const magSq = 1 + filterAlpha * filterAlpha - 2 * filterAlpha * Math.cos(omega)
        return 10 * Math.log10(Math.max(1e-6, magSq))
      })
      // Glottal roll-off: natural speech spectrum rolls off at approximately -6 dB / octave
      const glottalTilt = freqs.map((f) => -6 * Math.log2(f / 100))

      const highBoost = boostDb[boostDb.length - 1]

      return {
        x: freqs.map((f) => f / 1000), // in kHz
        y: boostDb,
        baseline: glottalTilt,
        stat1: `Pre-emphasis alpha = ${filterAlpha.toFixed(2)}`,
        stat2: `8 kHz HF boost: +${highBoost.toFixed(1)} dB`,
        stat3: 'Compensates -6 dB/oct tilt',
        warning:
          filterAlpha >= 0.95
            ? 'Optimal speech pre-emphasis: Equalizes formant spectral energy across low and high frequencies prior to STFT'
            : 'Mild pre-emphasis: Insufficient high-frequency boost leaves upper formants buried in quantization noise',
        axisXLabel: 'Acoustic Frequency (kHz)',
        axisYLabel: 'Filter Magnitude Response / Gain (dB)',
        xRange: [0.05, 8.2] as [number, number],
        yRange: [-45, 25] as [number, number],
      }
    } else if (visualizerMode === 'loudness_normalisation') {
      // 8 audio segments of varying acoustic intensity
      const segments = [1, 2, 3, 4, 5, 6, 7, 8]
      // Raw levels with outlier microphone click on segment 4 (0 dBFS peak, but quiet speech -35 dBFS)
      const rawPeaks = [-18, -24, -12, 0, -28, -15, -20, -32]
      // Peak-normalized levels: segment 4 was already at 0 dB, so all segments stay at raw levels!
      const peakNormalized = rawPeaks.map((p) => p)
      // LUFS / BS.1770 integrated perceptual normalization targets uniform -23 LUFS
      const lufsNormalized = segments.map(() => -23)

      return {
        x: segments,
        y: peakNormalized,
        baseline: lufsNormalized,
        stat1: 'Target level: -23.0 LUFS',
        stat2: 'Segment 4 contains transient pop (0 dBFS)',
        stat3: 'Peak Normalization Defeated',
        warning:
          'Single transient spikes fool peak normalization, leaving conversational speech quiet and inconsistent across files',
        axisXLabel: 'Audio Recording Instance Index',
        axisYLabel: 'Effective Segment Loudness (dBFS / LUFS)',
        xRange: [0.5, 8.5] as [number, number],
        yRange: [-40, 5] as [number, number],
      }
    } else {
      // VAD energy threshold trade-off: threshold from -50 to -15 dBFS
      const thresholds = [-50, -45, -40, -35, -30, -25, -20, -15]
      // Speech clipping rate (% of valid soft speech/consonants mistakenly dropped as silence)
      // High threshold (-15) causes high clipping (~45%), low threshold (-50) causes low clipping (~2%)
      const clippingRate = thresholds.map((th) => Math.max(1, Math.min(85, Math.pow(10, (th + 30) / 18) * 12)))
      // False speech alarm rate (% of background noise kept as speech)
      const falseAlarmRate = thresholds.map((th) => Math.max(1, Math.min(85, Math.pow(10, (-th - 30) / 18) * 12)))

      const currClip = Math.max(1, Math.min(85, Math.pow(10, (energyThresholdDb + 30) / 18) * 12))
      const currNoise = Math.max(1, Math.min(85, Math.pow(10, (-energyThresholdDb - 30) / 18) * 12))

      return {
        x: thresholds,
        y: clippingRate,
        baseline: falseAlarmRate,
        stat1: `Threshold: ${energyThresholdDb} dBFS`,
        stat2: `Speech clipping rate: ${currClip.toFixed(1)}%`,
        stat3: `Noise pass-through: ${currNoise.toFixed(1)}%`,
        warning:
          energyThresholdDb > -25
            ? 'Aggressive VAD: Drops quiet word onsets and unvoiced fricatives (/s/, /th/), corrupting ASR word error rate'
            : energyThresholdDb < -40
              ? 'Permissive VAD: Retains ambient background hiss and room reverb, wasting compute on empty frames'
              : 'Balanced VAD operating point: Preserves consonant boundaries while trimming empty pauses',
        axisXLabel: 'VAD Energy Cutoff Threshold (dBFS)',
        axisYLabel: 'Error Probability (%)',
        xRange: [-52, -13] as [number, number],
        yRange: [0, 90] as [number, number],
      }
    }
  }, [visualizerMode, filterAlpha, energyThresholdDb])

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
      title="Audio Preprocessing: Pre-Emphasis Tilting, Loudness Normalization, and VAD Cutoffs"
      state={state}
      hoverReadout={false}
      caption="Essential audio pipeline transforms: Compensating for natural glottal roll-off via pre-emphasis, the failure of peak normalization vs gated LUFS, and the speech clipping vs noise trade-off in VAD."
      readouts={
        <>
          <Readout label="Pipeline Parameter" value={results.stat1} />
          <Readout label="Acoustic Impact" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'preemphasis_tilt'
                ? 'Spectrum Balance'
                : visualizerMode === 'loudness_normalisation'
                  ? 'Consistency'
                  : 'Noise Leak'
            }
            value={results.stat3}
            color={visualizerMode === 'preemphasis_tilt' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="DSP Recommendation" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Audio Preprocessing Profile">
        <Curve
          name={
            visualizerMode === 'preemphasis_tilt'
              ? 'Glottal Acoustic Roll-off (-6 dB / octave)'
              : visualizerMode === 'loudness_normalisation'
                ? 'Integrated LUFS Reference (-23 LUFS)'
                : 'Background Noise Pass-Through Rate (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'preemphasis_tilt'
              ? 'Pre-Emphasis Filter Boost (dB)'
              : visualizerMode === 'loudness_normalisation'
                ? 'Peak Normalized Inconsistent Level'
                : 'Speech Clipping / Dropped Consonants Rate (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Operating Parameters" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

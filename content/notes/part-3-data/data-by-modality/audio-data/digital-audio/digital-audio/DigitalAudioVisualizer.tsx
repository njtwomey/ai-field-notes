import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DigitalAudioVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'nyquist_aliasing' | 'quantization_sqnr' | 'resampling_comparison'>(
      [
        {
          value: 'nyquist_aliasing',
          label: 'Nyquist-Shannon Sampling: Input Frequency vs Apparent Aliased Frequency (Shannon 1949)',
        },
        {
          value: 'quantization_sqnr',
          label: 'Quantization: Bit Depth B vs Signal-to-Quantization-Noise Ratio (Oppenheim 2010)',
        },
        {
          value: 'resampling_comparison',
          label: 'Downsampling: Anti-Aliasing Filter Cutoff vs High-Frequency Rejection',
        },
      ],
      'nyquist_aliasing',
      { label: 'digital audio property' },
    ),
    sampleRateKhz: slider(8, 48, 16, {
      step: 4,
      label: 'sampling rate f_s (kHz)',
    }),
    bitDepth: slider(4, 16, 16, {
      step: 2,
      label: 'quantization bit depth B (bits)',
    }),
  })

  const { visualizerMode, sampleRateKhz, bitDepth } = state

  const results = useMemo(() => {
    if (visualizerMode === 'nyquist_aliasing') {
      // Input frequencies from 1 kHz to 30 kHz
      const freqs = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30]
      const nyquistLimit = sampleRateKhz / 2
      // Apparent frequency: if f <= nyquist, f. If f > nyquist, reflects back as |f - round(f/fs)*fs|
      const apparentFreqs = freqs.map((f) => {
        if (f <= nyquistLimit) return f
        // Fold-back alias in first Nyquist zone
        const rem = f % sampleRateKhz
        return rem <= nyquistLimit ? rem : sampleRateKhz - rem
      })
      const nyquistBaseline = freqs.map(() => nyquistLimit)

      const testInput = 14
      const testRem = testInput % sampleRateKhz
      const testAliased =
        testInput <= nyquistLimit ? testInput : testRem <= nyquistLimit ? testRem : sampleRateKhz - testRem

      return {
        x: freqs,
        y: apparentFreqs,
        baseline: nyquistBaseline,
        stat1: `Sampling rate f_s = ${sampleRateKhz} kHz`,
        stat2: `Nyquist ceiling: ${nyquistLimit.toFixed(1)} kHz`,
        stat3:
          testInput > nyquistLimit ? `14 kHz aliases to ${testAliased.toFixed(1)} kHz` : 'Clean capture (< Nyquist)',
        warning:
          sampleRateKhz <= 16
            ? 'Speech standard (16 kHz): Frequencies above 8 kHz alias into the audible speech spectrum without analog low-pass filtering'
            : 'Wideband audio: Captures high harmonics up to 24 kHz without spectral aliasing',
        axisXLabel: 'Physical Acoustic Input Frequency (kHz)',
        axisYLabel: 'Apparent Digitized Frequency (kHz)',
        xRange: [0.5, 30.5] as [number, number],
        yRange: [0, Math.max(...apparentFreqs, nyquistLimit) * 1.15] as [number, number],
      }
    } else if (visualizerMode === 'quantization_sqnr') {
      // Bit depths B from 2 to 16 bits
      const bits = [2, 4, 6, 8, 10, 12, 14, 16]
      // SQNR = 6.02 * B + 1.76 dB
      const sqnr = bits.map((b) => 6.02 * b + 1.76)
      // CD Audio benchmark: 16-bit = 98.08 dB
      const cdBenchmark = bits.map(() => 98.08)

      const currSQNR = 6.02 * bitDepth + 1.76
      const quantLevels = Math.pow(2, bitDepth)

      return {
        x: bits,
        y: sqnr,
        baseline: cdBenchmark,
        stat1: `Bit depth B = ${bitDepth} bits`,
        stat2: `Discrete levels: ${quantLevels.toLocaleString()}`,
        stat3: `SQNR: ${currSQNR.toFixed(1)} dB`,
        warning:
          bitDepth <= 8
            ? 'Severe quantization noise: 8-bit audio exhibits noticeable audible hiss and harmonic distortion'
            : 'High fidelity: 16-bit PCM dynamic range (>96 dB) preserves whisper-level details and quiet speech',
        axisXLabel: 'Quantization Bit Depth B (Bits / Sample)',
        axisYLabel: 'Theoretical SQNR (dB Dynamic Range)',
        xRange: [1.5, 16.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else {
      // Downsampling filter attenuation curve across frequency (normalized to 0 to 1 pi)
      const freqs = [0, 2, 4, 6, 7.5, 8, 9, 10, 12, 14, 16]
      const targetNyquist = sampleRateKhz / 2
      // Polyphase anti-aliasing filter attenuation (dB): 0 dB in passband, sharp drop to -80 dB past cutoff
      const attenuation = freqs.map((f) => {
        if (f < targetNyquist * 0.9) return 0
        if (f <= targetNyquist) return -((f - targetNyquist * 0.9) / (targetNyquist * 0.1)) * 40
        return -80
      })
      // Unfiltered naive decimation pass-through (0 dB flat across all freqs, aliasing everything)
      const naiveDecimation = freqs.map(() => 0)

      return {
        x: freqs,
        y: attenuation,
        baseline: naiveDecimation,
        stat1: `Target rate f_s = ${sampleRateKhz} kHz`,
        stat2: `Anti-aliasing cutoff: ${(targetNyquist * 0.9).toFixed(1)} kHz`,
        stat3: 'Stopband rejection: -80 dB',
        warning:
          'Naive decimation passes all frequencies without attenuation, folding out-of-band noise directly into speech features',
        axisXLabel: 'Acoustic Frequency (kHz)',
        axisYLabel: 'Filter Magnitude Response (dB)',
        xRange: [-0.5, 16.5] as [number, number],
        yRange: [-85, 5] as [number, number],
      }
    }
  }, [visualizerMode, sampleRateKhz, bitDepth])

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
      title="Digital Audio: Nyquist Aliasing, Quantization SQNR, and Resampling Dynamics"
      state={state}
      hoverReadout={false}
      caption="Fundamentals of digital audio discretization: Frequency folding past the Nyquist ceiling, dynamic range scaling with quantization bit depth, and anti-aliasing filter rejection during rate conversion."
      readouts={
        <>
          <Readout label="Audio Format" value={results.stat1} />
          <Readout label="Resolution Metric" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'nyquist_aliasing'
                ? 'Aliasing Status'
                : visualizerMode === 'quantization_sqnr'
                  ? 'Dynamic Range'
                  : 'Filter Quality'
            }
            value={results.stat3}
            color={visualizerMode === 'nyquist_aliasing' ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="DSP Guidance" value={results.warning} color="var(--red-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Digital Audio Representation Profile">
        <Curve
          name={
            visualizerMode === 'nyquist_aliasing'
              ? 'Nyquist Ceiling Reference (f_s / 2)'
              : visualizerMode === 'quantization_sqnr'
                ? '16-Bit CD Audio Benchmark (98 dB)'
                : 'Naive Decimation (No Anti-Aliasing)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'nyquist_aliasing'
              ? 'Digitized Apparent Frequency Trajectory'
              : visualizerMode === 'quantization_sqnr'
                ? 'Quantization SQNR Scaling (6.02 B + 1.76 dB)'
                : 'Polyphase Anti-Aliasing Filter Response'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Spectral Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

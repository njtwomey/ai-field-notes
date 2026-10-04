import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'

const bark = (f: number) => 13 * Math.atan(0.00076 * f) + 3.5 * Math.atan((f / 7500) ** 2)
/** Threshold in quiet, dB SPL (Terhardt's approximation). */
const quiet = (f: number) => {
  const k = f / 1000
  return 3.64 * k ** -0.8 - 6.5 * Math.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4
}
/** Schroeder's spreading function, dB, as a function of the Bark distance from the masker. */
const spread = (dz: number) => 15.81 + 7.5 * (dz + 0.474) - 17.5 * Math.sqrt(1 + (dz + 0.474) ** 2)

const FREQS = Array.from({ length: 400 }, (_, i) => 20 * 1000 ** (i / 399)) // 20 Hz to 20 kHz, log-spaced

/**
 * The masking threshold of one or two tones: each tone's level, spread over neighbouring critical bands and lowered by
 * a masking offset, combined with the threshold in quiet. Components below the curve are inaudible, so the quantisation
 * noise of a perceptual coder can be shaped to stay under it.
 */
export function MaskingThreshold() {
  const state = useFigureState({
    f1: int(1000, { min: 100, max: 8000, step: 10, label: 'tone 1 frequency (Hz)' }),
    l1: int(70, { min: 20, max: 90, step: 1, label: 'tone 1 level (dB SPL)' }),
    f2: int(4000, { min: 100, max: 12000, step: 10, label: 'tone 2 frequency (Hz)' }),
    l2: int(0, { min: 0, max: 90, step: 1, label: 'tone 2 level (dB SPL, 0 = off)' }),
    offset: int(10, { min: 0, max: 25, step: 1, label: 'masking offset (dB)' }),
  })

  const r = useMemo(() => {
    const maskers = [
      [state.f1, state.l1],
      [state.f2, state.l2],
    ].filter(([, l]) => l > 0)
    // Powers add: combine the thresholds in the power domain, then return to dB.
    const threshold = FREQS.map((f) => {
      const power = maskers.reduce(
        (s, [fm, lm]) => s + 10 ** ((lm + spread(bark(f) - bark(fm)) - state.offset) / 10),
        10 ** (quiet(f) / 10),
      )
      return 10 * Math.log10(power)
    })
    return { threshold, maskers }
  }, [state.f1, state.l1, state.f2, state.l2, state.offset])

  const logF = FREQS.map((f) => Math.log10(f))
  const series = [
    { name: 'threshold in quiet', x: logF, y: FREQS.map(quiet), dashed: true, slot: 2 },
    { name: 'global masking threshold', x: logF, y: r.threshold, slot: 1 },
    {
      name: 'masking tones',
      x: r.maskers.map(([f]) => Math.log10(f)),
      y: r.maskers.map(([, l]) => l),
      emphasis: true,
    },
  ] as const
  const smr = state.l1 - r.threshold[FREQS.findIndex((f) => f >= state.f1)]

  const xAxis = useAxis({ label: 'log₁₀ frequency (Hz)', range: [Math.log10(20), Math.log10(20000)] })
  const yAxis = useAxis({ label: 'dB SPL', range: [-10, 100] })
  return (
    <Figure
      title="Simultaneous masking"
      state={state}
      caption="The threshold in quiet (dashed) and the global masking threshold (solid) with one or two masking tones. Each tone raises the threshold near it, spread over neighbouring critical bands with Schroeder's spreading function in Bark and lowered by a masking offset. The spread is asymmetric: masking reaches much further towards high frequencies than towards low ones. Any component, including quantisation noise, below the solid curve is inaudible. The horizontal axis is log₁₀ of frequency in Hz."

      readouts={
        <>
          <Readout label="tone 1 at" value={`${formatNumber(bark(state.f1))} Bark`} />
          <Readout label="signal-to-mask ratio at tone 1" value={`${formatNumber(smr)} dB`} />
          <Readout label="bits for noise below mask ≈ SMR / 6.02" value={formatNumber(Math.max(0, smr / 6.02))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Points {...series[2]} />
      </Plot>
    </Figure>
  )
}

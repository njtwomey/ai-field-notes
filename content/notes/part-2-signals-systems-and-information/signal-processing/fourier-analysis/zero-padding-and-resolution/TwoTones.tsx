import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { rfft } from 'aifn-compute/foundation/fourier'
import { complexAbs, toFlat } from 'aifn-compute/foundation/tensor'

const FS = 1000
const F1 = 100

/** |X[k]|, k = 0..size/2, of x zero-padded to `size` samples. */
const magnitudeSpectrum = (x: number[], size: number) => toFlat(complexAbs(rfft(x, { n: size })))
/** The frequencies in Hz of the one-sided bins of a `size`-point DFT. */
const binFrequencies = (size: number) => Array.from({ length: size / 2 + 1 }, (_, k) => (k * FS) / size)

/**
 * Two tones Δf apart, observed for N samples. Zero-padding samples the same DTFT more finely; only a longer
 * observation separates the peaks.
 */
export function TwoTones() {
  const state = useFigureState({
    n: int(100, { min: 40, max: 400, step: 20, label: 'samples N', format: (v) => String(v) }),
    df: float(5, { min: 1, max: 30, step: 0.5, label: 'tone spacing Δf (Hz)' }),
    pad: int(1, { min: 1, max: 16, step: 1, label: 'zero-padding factor', format: (v) => `${v}×` }),
  })

  const r = useMemo(() => {
    const x = Array.from({ length: state.n }, (_, k) => {
      const t = k / FS
      return Math.cos(2 * Math.PI * F1 * t) + Math.cos(2 * Math.PI * (F1 + state.df) * t + 1)
    })
    const size = 2 ** Math.ceil(Math.log2(state.n * state.pad))
    const mag = magnitudeSpectrum(x, size).map((m) => m / state.n)
    // A reference curve: the same signal padded 64 times, i.e. essentially the DTFT.
    const denseSize = 2 ** Math.ceil(Math.log2(state.n * 64))
    const dense = magnitudeSpectrum(x, denseSize).map((m) => m / state.n)
    return { mag, freqs: binFrequencies(size), dense, denseFreqs: binFrequencies(denseSize) }
  }, [state.n, state.df, state.pad])

  const view = (freqs: number[], ys: number[]) => {
    const keep = freqs.map((f, i) => [f, ys[i]] as const).filter(([f]) => f >= 60 && f <= 160)
    return { x: keep.map(([f]) => f), y: keep.map(([, v]) => v) }
  }
  const dtft = view(r.denseFreqs, r.dense)
  const dft = view(r.freqs, r.mag)
  const series = [
    { name: 'DTFT of the N samples', x: dtft.x, y: dtft.y, slot: 0 },
    { name: 'DFT bins after padding', x: dft.x, y: dft.y, emphasis: true },
  ] as const
  const resolution = FS / state.n

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [60, 160] })
  const yAxis = useAxis({ label: 'magnitude', hold: 'union' })
  return (
    <Figure
      title="Zero-padding is not resolution"
      state={state}
      caption="Two cosines, at 100 Hz and 100 + Δf Hz, sampled at 1 kHz for N samples. The line is the DTFT of those N samples; the points are the DFT after padding with zeros. More padding puts more points on the same curve. Only a longer observation, larger N, narrows the peaks enough to separate the tones, roughly once Δf exceeds f_s/N."

      readouts={
        <>
          <Readout label="resolution f_s/N" value={`${formatNumber(resolution)} Hz`} />
          <Readout label="bin spacing" value={`${formatNumber(FS / (r.freqs.length * 2 - 2))} Hz`} />
          <Readout label="Δf vs f_s/N" value={state.df >= resolution ? 'separable' : 'too close'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
      </Plot>
    </Figure>
  )
}

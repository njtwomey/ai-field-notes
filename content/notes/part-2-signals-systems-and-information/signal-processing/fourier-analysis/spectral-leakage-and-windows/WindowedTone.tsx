import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { rfft } from 'aifn/foundation/fourier'
import { complexAbs, toFlat } from 'aifn/foundation/tensor'
import { getWindow } from 'aifn/signal/windows'

const N = 64
const PAD = 1024
type Taper = 'rectangular' | 'hann' | 'hamming' | 'blackman' | 'kaiser'

/** Measured for N = 64 periodic windows (see the note's table). */
const PROPERTIES: Record<Taper, { lobe: number; sidelobe: number; scallop: number; enbw: number }> = {
  rectangular: { lobe: 2, sidelobe: -13.3, scallop: 3.92, enbw: 1.0 },
  hann: { lobe: 4, sidelobe: -31.5, scallop: 1.42, enbw: 1.5 },
  hamming: { lobe: 4, sidelobe: -42.4, scallop: 1.75, enbw: 1.36 },
  blackman: { lobe: 6, sidelobe: -58.1, scallop: 1.1, enbw: 1.73 },
  kaiser: { lobe: 5.83, sidelobe: -63.4, scallop: 1.11, enbw: 1.72 },
}

/** Periodic (DFT-even) window of length N; Kaiser with β = 8.6. */
const taper = (name: Taper) => toFlat(getWindow(name === 'kaiser' ? { name, beta: 8.6 } : name, N, { periodic: true }))

/** |X[k]|, k = 0..size/2, of x zero-padded to `size` samples. */
const magnitudeSpectrum = (x: number[], size: number) => toFlat(complexAbs(rfft(x, { n: size })))
/** Decibels, 20 log₁₀ of a magnitude, floored at −120 dB. */
const db = (m: number) => Math.max(-120, 20 * Math.log10(Math.max(m, 1e-300)))

/**
 * A tone at a chosen frequency, in DFT bins, analysed with a 64-point window. Off-bin frequencies spread energy into
 * every bin; the window trades main-lobe width against sidelobe level.
 */
export function WindowedTone() {
  const state = useFigureState({
    taper: choice<Taper>(
      [
        { value: 'rectangular', label: 'rectangular' },
        { value: 'hann', label: 'Hann' },
        { value: 'hamming', label: 'Hamming' },
        { value: 'blackman', label: 'Blackman' },
        { value: 'kaiser', label: 'Kaiser β=8.6' },
      ],
      'rectangular',
      { label: 'window' },
    ),
    bin: slider(2, 28, 10.5, { step: 0.05, label: 'tone frequency (bins)' }),
  })
  const { taper: name, bin } = state

  const r = useMemo(() => {
    const w = taper(name)
    const sum = w.reduce((s, v) => s + v, 0)
    const x = Array.from({ length: N }, (_, n) => Math.cos((2 * Math.PI * bin * n) / N) * w[n])
    // Scale so a tone exactly on a bin reads 0 dB: the coherent gain of a cosine is sum(w)/2.
    const spectrum = magnitudeSpectrum(x, PAD).map((m) => db(m / (sum / 2)))
    const coarse = magnitudeSpectrum(x, N).map((m) => db(m / (sum / 2)))
    return { w, spectrum, coarse, peak: Math.max(...coarse) }
  }, [name, bin])

  const fine = r.spectrum.map((_, k) => (k * N) / PAD)
  const series = [
    { name: 'DTFT of the windowed tone', x: fine, y: r.spectrum, slot: 0 },
    { name: '64-point DFT bins', x: r.coarse.map((_, k) => k), y: r.coarse, emphasis: true },
  ] as const
  const time = [{ name: 'window', x: Array.from(r.w, (_, n) => n), y: Array.from(r.w), slot: 1 }] as const
  const props = PROPERTIES[name]

  const xAxis = useAxis({ label: 'frequency (bins)', range: [0, 32] })
  const yAxis = useAxis({ label: 'dB', range: [-120, 5] })
  const xAxis2 = useAxis({ label: 'n', range: [0, N - 1] })
  const yAxis2 = useAxis({ label: 'w[n]', range: [0, 1.05] })
  return (
    <Figure
      title="Leakage and windows"
      caption="A cosine analysed with 64 samples. When its frequency falls exactly on a bin, the DFT (points) shows one clean line; anywhere between bins, the window's own spectrum (line) is sampled off its peak and energy leaks into every bin. Tapered windows push the far leakage down by tens of decibels at the price of a wider main lobe. Drag the tone or use the slider."
      state={state}
      readouts={
        <>
          <Readout label="main lobe (null to null)" value={`${formatNumber(props.lobe)} bins`} />
          <Readout label="peak sidelobe" value={`${props.sidelobe} dB`} />
          <Readout label="scalloping loss" value={`${props.scallop} dB`} />
          <Readout label="ENBW" value={`${props.enbw} bins`} />
          <Readout label="largest bin now" value={`${formatNumber(r.peak)} dB`} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...series[0]} />
          <Points {...series[1]} />
          <Handle {...state.handle('bin', { label: 'tone' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={140}>
          <Curve {...time[0]} />
        </Plot>
      </div>
    </Figure>
  )
}

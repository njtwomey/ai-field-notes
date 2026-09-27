import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
// Imported under another name: a module-level `window` would shadow the browser global that React's refresh checks.
import { db, kaiser, magnitudeSpectrum, makeWindow as windowFunction, type WindowName } from '@/lib/dsp'

const N = 64
const PAD = 1024
type Choice = WindowName | 'kaiser'

/** Measured for N = 64 periodic windows (see the note's table). */
const PROPERTIES: Record<Choice, { lobe: number; sidelobe: number; scallop: number; enbw: number }> = {
  rectangular: { lobe: 2, sidelobe: -13.3, scallop: 3.92, enbw: 1.0 },
  hann: { lobe: 4, sidelobe: -31.5, scallop: 1.42, enbw: 1.5 },
  hamming: { lobe: 4, sidelobe: -42.4, scallop: 1.75, enbw: 1.36 },
  blackman: { lobe: 6, sidelobe: -58.1, scallop: 1.1, enbw: 1.73 },
  kaiser: { lobe: 5.83, sidelobe: -63.4, scallop: 1.11, enbw: 1.72 },
}

/** Periodic (DFT-even) window of length N; Kaiser with β = 8.6. */
const taper = (choice: Choice) =>
  choice === 'kaiser' ? kaiser(N + 1, 8.6).slice(0, N) : windowFunction(choice, N, true)

/**
 * A tone at a chosen frequency, in DFT bins, analysed with a 64-point window. Off-bin frequencies spread energy into
 * every bin; the window trades main-lobe width against sidelobe level.
 */
export function WindowedTone() {
  const [choice, setChoice] = useState<Choice>('rectangular')
  const bin = useParam(10.5, { min: 2, max: 28, step: 0.05 })

  const r = useMemo(() => {
    const w = taper(choice)
    const sum = w.reduce((s, v) => s + v, 0)
    const x = Array.from({ length: N }, (_, n) => Math.cos((2 * Math.PI * bin.value * n) / N) * w[n])
    // Scale so a tone exactly on a bin reads 0 dB: the coherent gain of a cosine is sum(w)/2.
    const spectrum = Array.from(magnitudeSpectrum(x, PAD), (m) => db(m / (sum / 2), -120))
    const coarse = Array.from(magnitudeSpectrum(x, N), (m) => db(m / (sum / 2), -120))
    return { w, spectrum, coarse, peak: Math.max(...coarse) }
  }, [choice, bin.value])

  const fine = r.spectrum.map((_, k) => (k * N) / PAD)
  const series: XYSeries[] = [
    { name: 'DTFT of the windowed tone', type: 'line', x: fine, y: r.spectrum, slot: 0 },
    { name: '64-point DFT bins', type: 'scatter', x: r.coarse.map((_, k) => k), y: r.coarse, emphasis: true },
  ]
  const time: XYSeries[] = [
    { name: 'window', type: 'line', x: Array.from(r.w, (_, n) => n), y: Array.from(r.w), slot: 1 },
  ]
  const handles: Handle[] = [{ kind: 'x', at: bin.value, label: 'tone', onDrag: (v) => bin.set(v) }]
  const props = PROPERTIES[choice]

  return (
    <Interactive
      title="Leakage and windows"
      caption="A cosine analysed with 64 samples. When its frequency falls exactly on a bin, the DFT (points) shows one clean line; anywhere between bins, the window's own spectrum (line) is sampled off its peak and energy leaks into every bin. Tapered windows push the far leakage down by tens of decibels at the price of a wider main lobe. Drag the tone or use the slider."
      controls={
        <>
          <ParamChoice
            label="window"
            value={choice}
            onChange={setChoice}
            options={[
              { value: 'rectangular', label: 'rectangular' },
              { value: 'hann', label: 'Hann' },
              { value: 'hamming', label: 'Hamming' },
              { value: 'blackman', label: 'Blackman' },
              { value: 'kaiser', label: 'Kaiser β=8.6' },
            ]}
          />
          <ParamSlider label="tone frequency (bins)" param={bin} />
        </>
      }
      readout={
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
        <XYChart
          series={series}
          xLabel="frequency (bins)"
          yLabel="dB"
          xRange={[0, 32]}
          yRange={[-120, 5]}
          handles={handles}
          height={280}
        />
        <XYChart series={time} xLabel="n" yLabel="w[n]" xRange={[0, N - 1]} yRange={[0, 1.05]} height={140} />
      </div>
    </Interactive>
  )
}

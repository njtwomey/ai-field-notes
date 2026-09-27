import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { fft, makeWindow as dspWindow } from '@/lib/dsp'

type Shape = 'gaussian' | 'hann' | 'blackman'

const FS = 1000
const LENGTH = 1024
const NFFT = 16384
const BOUND = 1 / (4 * Math.PI)

/** Window samples centred in LENGTH, for width parameter s (ms): Gaussian σ = s; Hann and Blackman span 6s. */
function makeWindow(shape: Shape, s: number): Float64Array {
  const w = new Float64Array(LENGTH)
  const mid = (LENGTH - 1) / 2
  if (shape === 'gaussian') {
    const sigma = (s / 1000) * FS
    for (let i = 0; i < LENGTH; i++) w[i] = Math.exp(-((i - mid) ** 2) / (2 * sigma * sigma))
    return w
  }
  const span = Math.min(LENGTH, Math.round((6 * s * FS) / 1000))
  const core = dspWindow(shape, span)
  const start = Math.floor((LENGTH - span) / 2)
  w.set(core, start)
  return w
}

/**
 * The spread of a window in time and in frequency, each measured as the standard deviation of its normalised energy
 * density. Their product is at least 1/(4π), with equality only for the Gaussian.
 */
export function GaborBox() {
  const [shape, setShape] = useState<Shape>('gaussian')
  const width = useParam(20, { min: 4, max: 80, step: 1 })

  const r = useMemo(() => {
    const w = makeWindow(shape, width.value)
    const mid = (LENGTH - 1) / 2
    const energy = w.reduce((a, v) => a + v * v, 0)
    const sigmaT = Math.sqrt(w.reduce((a, v, i) => a + ((i - mid) / FS) ** 2 * v * v, 0) / energy)
    const { re, im } = fft(w, NFFT)
    const power = re.map((v, k) => v * v + im[k] * im[k])
    const freq = (k: number) => ((k <= NFFT / 2 ? k : k - NFFT) * FS) / NFFT
    const total = power.reduce((a, v) => a + v, 0)
    const sigmaF = Math.sqrt(power.reduce((a, v, k) => a + freq(k) ** 2 * v, 0) / total)
    const peak = Math.sqrt(Math.max(...power))
    const timeX = Array.from(w, (_, i) => ((i - mid) / FS) * 1000)
    // Show ±6σ_f of the spectrum at 601 points, each read from the nearest FFT bin.
    const span = Math.max(6 * sigmaF, 4)
    const specF = Array.from({ length: 601 }, (_, i) => -span + (2 * span * i) / 600)
    const bin = (f: number) => (Math.round((f * NFFT) / FS) + NFFT) % NFFT
    return {
      time: { x: timeX, y: Array.from(w) },
      spec: { x: specF, y: specF.map((f) => Math.sqrt(power[bin(f)]) / peak) },
      timeSpan: Math.min(512, 3.5 * width.value),
      freqSpan: span,
      sigmaT,
      sigmaF,
    }
  }, [shape, width.value])

  return (
    <Interactive
      title="A window cannot be narrow in time and frequency at once"
      caption="Left: the window in time. Right: its magnitude spectrum. The readouts measure each spread as the standard deviation of the normalised energy, |w(t)|² in time and |W(f)|² in frequency. Narrowing the window in time widens it in frequency, and the product σ_t σ_f never falls below 1/(4π) ≈ 0.0796. The Gaussian meets the bound exactly at every width; Hann and Blackman windows, shown spanning six widths, come within a few per cent."
      controls={
        <>
          <ParamChoice
            label="window"
            value={shape}
            onChange={setShape}
            options={[
              { value: 'gaussian', label: 'Gaussian' },
              { value: 'hann', label: 'Hann' },
              { value: 'blackman', label: 'Blackman' },
            ]}
          />
          <ParamSlider label="width (ms)" param={width} format={(v) => `${v} ms`} />
        </>
      }
      readout={
        <>
          <Readout label="σ_t" value={`${formatNumber(1000 * r.sigmaT)} ms`} />
          <Readout label="σ_f" value={`${formatNumber(r.sigmaF)} Hz`} />
          <Readout label="σ_t σ_f" value={formatNumber(r.sigmaT * r.sigmaF)} />
          <Readout label="ratio to 1/(4π)" value={formatNumber((r.sigmaT * r.sigmaF) / BOUND)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={[{ name: 'w(t)', type: 'line', ...r.time, slot: 0, area: true }]}
          xLabel="time (ms)"
          yLabel="amplitude"
          xRange={[-r.timeSpan, r.timeSpan]}
          yRange={[0, 1.05]}
          height={260}
        />
        <XYChart
          series={[{ name: '|W(f)|', type: 'line', ...r.spec, slot: 1, area: true }]}
          xLabel="frequency (Hz)"
          yLabel="relative magnitude"
          xRange={[-r.freqSpan, r.freqSpan]}
          yRange={[0, 1.05]}
          height={260}
        />
      </div>
    </Interactive>
  )
}

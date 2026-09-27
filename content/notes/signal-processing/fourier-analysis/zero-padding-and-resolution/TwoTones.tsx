import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { binFrequencies, magnitudeSpectrum } from '@/lib/dsp'

const FS = 1000
const F1 = 100

/**
 * Two tones Δf apart, observed for N samples. Zero-padding samples the same DTFT more finely; only a longer
 * observation separates the peaks.
 */
export function TwoTones() {
  const n = useParam(100, { min: 40, max: 400, step: 20 })
  const df = useParam(5, { min: 1, max: 30, step: 0.5 })
  const pad = useParam(1, { min: 1, max: 16, step: 1 })

  const r = useMemo(() => {
    const x = Array.from({ length: n.value }, (_, k) => {
      const t = k / FS
      return Math.cos(2 * Math.PI * F1 * t) + Math.cos(2 * Math.PI * (F1 + df.value) * t + 1)
    })
    const size = 2 ** Math.ceil(Math.log2(n.value * pad.value))
    const mag = Array.from(magnitudeSpectrum(x, size), (m) => m / n.value)
    // A reference curve: the same signal padded 64 times, i.e. essentially the DTFT.
    const dense = Array.from(magnitudeSpectrum(x, 2 ** Math.ceil(Math.log2(n.value * 64))), (m) => m / n.value)
    return { mag, freqs: binFrequencies(size, FS), dense, denseFreqs: binFrequencies(dense.length * 2 - 2, FS) }
  }, [n.value, df.value, pad.value])

  const view = (freqs: number[], ys: number[]) => {
    const keep = freqs.map((f, i) => [f, ys[i]] as const).filter(([f]) => f >= 60 && f <= 160)
    return { x: keep.map(([f]) => f), y: keep.map(([, v]) => v) }
  }
  const dtft = view(r.denseFreqs, r.dense)
  const dft = view(r.freqs, r.mag)
  const series: XYSeries[] = [
    { name: 'DTFT of the N samples', type: 'line', x: dtft.x, y: dtft.y, slot: 0 },
    { name: 'DFT bins after padding', type: 'scatter', x: dft.x, y: dft.y, emphasis: true },
  ]
  const resolution = FS / n.value

  return (
    <Interactive
      title="Zero-padding is not resolution"
      caption="Two cosines, at 100 Hz and 100 + Δf Hz, sampled at 1 kHz for N samples. The line is the DTFT of those N samples; the points are the DFT after padding with zeros. More padding puts more points on the same curve. Only a longer observation, larger N, narrows the peaks enough to separate the tones, roughly once Δf exceeds f_s/N."
      controls={
        <>
          <ParamSlider label="samples N" param={n} format={(v) => String(v)} withArrows />
          <ParamSlider label="tone spacing Δf (Hz)" param={df} />
          <ParamSlider label="zero-padding factor" param={pad} format={(v) => `${v}×`} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="resolution f_s/N" value={`${formatNumber(resolution)} Hz`} />
          <Readout label="bin spacing" value={`${formatNumber(FS / (r.freqs.length * 2 - 2))} Hz`} />
          <Readout label="Δf vs f_s/N" value={df.value >= resolution ? 'separable' : 'too close'} />
        </>
      }
    >
      <XYChart series={series} xLabel="frequency (Hz)" yLabel="magnitude" xRange={[60, 160]} height={280} />
    </Interactive>
  )
}

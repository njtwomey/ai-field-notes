import { useMemo } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, formatNumber, useParam } from 'aifn-render'
import { binFrequencies, db, stft } from '@/lib/dsp'
import { chirp } from '../_shared/tf'

const FS = 8000
const N = 4096
const FLOOR = 70

// A chirp from 300 Hz to 3.5 kHz over the whole signal, two tones 100 Hz apart in the first half, and a click.
const SIGNAL = (() => {
  const x = chirp(N, FS, 300, 3500, 0.6)
  for (let n = 0; n < N / 2; n++) {
    const t = n / FS
    x[n] += 0.5 * Math.cos(2 * Math.PI * 1200 * t) + 0.5 * Math.cos(2 * Math.PI * 1300 * t)
  }
  x[Math.round(0.4 * FS)] += 12
  return x
})()

/**
 * Spectrogram of a chirp, a close pair of tones and a click, with the window length as the control. Short windows
 * locate the click and follow the chirp; long windows separate the two tones.
 */
export function StftExplorer() {
  const exponent = useParam(8, { min: 5, max: 10, step: 1 })
  const size = 2 ** exponent.value
  const hop = size / 4

  const r = useMemo(() => {
    const { frames, centres } = stft(SIGNAL, size, hop, 'hann', size)
    const freqs = binFrequencies(size, FS)
    let peak = 0
    for (const f of frames) for (const v of f) peak = Math.max(peak, v)
    // Rows are frequency bins, columns frames; dB relative to the loudest cell, floored at −FLOOR dB.
    const z = freqs.map((_, k) => frames.map((f) => db(f[k] / peak, -FLOOR)))
    return { z, times: centres.map((c) => c / FS), freqs }
  }, [size, hop])

  return (
    <Interactive
      title="The window length trades time for frequency"
      caption="A chirp sweeping from 300 Hz to 3.5 kHz, two tones at 1.2 and 1.3 kHz in the first half, and a click at 0.4 s, sampled at 8 kHz. Each column is the magnitude spectrum of one Hann-windowed frame, with frames every quarter window. A short window places the click in a thin vertical line but smears the two tones into one band; a long window separates the tones and blurs the click and the chirp. No single length shows everything sharply."
      controls={
        <ParamSlider label="window length (samples)" param={exponent} format={(v) => String(2 ** v)} withArrows />
      }
      readout={
        <>
          <Readout label="window duration" value={`${formatNumber((1000 * size) / FS)} ms`} />
          <Readout label="bin spacing f_s/L" value={`${formatNumber(FS / size)} Hz`} />
          <Readout label="Hann main lobe, null to null" value={`${formatNumber((4 * FS) / size)} Hz`} />
          <Readout label="frames" value={r.times.length} />
        </>
      }
    >
      <Heatmap
        x={r.times}
        y={r.freqs}
        z={r.z}
        range={[-FLOOR, 0]}
        xLabel="time (s)"
        yLabel="frequency (Hz)"
        valueLabel="dB"
        height={380}
      />
    </Interactive>
  )
}

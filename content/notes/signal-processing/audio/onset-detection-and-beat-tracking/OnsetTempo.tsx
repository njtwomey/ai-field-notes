import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { stft } from '@/lib/dsp'
import { rng } from '@/lib/math'

const FS = 8000
const SECONDS = 4
const SIZE = 256
const HOP = 64
const RATE = FS / HOP // onset-envelope frames per second

/** Decaying harmonic notes on every beat at the given tempo, random pitches, with timing jitter and noise. */
function notes(bpm: number, jitterMs: number, noise: number, seed: number) {
  const g = rng(seed)
  const x = new Array(FS * SECONDS).fill(0)
  const beat = 60 / bpm
  const onsets: number[] = []
  for (let t = 0.1; t < SECONDS - 0.2; t += beat) {
    const start = t + ((g.uniform() - 0.5) * 2 * jitterMs) / 1000
    onsets.push(start)
    const f0 = 150 + 400 * g.uniform()
    const n0 = Math.round(start * FS)
    for (let i = 0; i < 0.25 * FS && n0 + i < x.length; i++) {
      const env = Math.exp(-i / (0.06 * FS))
      let s = 0
      for (let h = 1; h <= 6; h++) s += Math.cos((2 * Math.PI * h * f0 * i) / FS) / h
      x[n0 + i] += env * s
    }
  }
  return { x: x.map((v) => v + noise * g.normal()), onsets }
}

/**
 * Spectral-flux onset detection and tempo estimation. The onset envelope is the half-wave-rectified increase of the
 * log-compressed magnitude spectrum between frames; peaks above a moving median are onsets, and the autocorrelation of
 * the envelope peaks at the beat period.
 */
export function OnsetTempo() {
  const bpm = useParam(110, { min: 60, max: 180, step: 1 })
  const jitter = useParam(10, { min: 0, max: 60, step: 1 })
  const noise = useParam(0.05, { min: 0, max: 0.5, step: 0.01 })
  const seed = useParam(2, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const { x, onsets } = notes(bpm.value, jitter.value, noise.value, seed.value)
    const { frames, centres } = stft(x, SIZE, HOP)
    const logMag = frames.map((f) => Array.from(f, (v) => Math.log(1 + 10 * v)))
    const flux = logMag.map((f, t) => (t === 0 ? 0 : f.reduce((s, v, k) => s + Math.max(0, v - logMag[t - 1][k]), 0)))
    // Adaptive threshold: moving median over ±0.1 s plus a margin; peaks must be local maxima above it.
    const half = Math.round(0.1 * RATE)
    const margin = 0.1 * Math.max(...flux)
    const threshold = flux.map((_, t) => {
      const w = flux.slice(Math.max(0, t - half), t + half + 1).sort((a, b) => a - b)
      return w[Math.floor(w.length / 2)] + margin
    })
    const picked = flux.flatMap((v, t) =>
      t > 0 && t < flux.length - 1 && v > threshold[t] && v >= flux[t - 1] && v > flux[t + 1] ? [t] : [],
    )
    // Tempo: autocorrelation of the mean-removed envelope over lags for 60–200 BPM.
    const mean = flux.reduce((a, b) => a + b, 0) / flux.length
    const e = flux.map((v) => v - mean)
    const lagMin = Math.round((60 / 200) * RATE)
    const lagMax = Math.round((60 / 55) * RATE)
    const acf = Array.from({ length: lagMax + 1 }, (_, L) =>
      e.reduce((s, v, t) => (t + L < e.length ? s + v * e[t + L] : s), 0),
    )
    let best = lagMin
    for (let L = lagMin; L <= lagMax; L++) if (acf[L] > acf[best]) best = L
    const times = centres.map((c) => c / FS)
    const hits = onsets.filter((o) => picked.some((t) => Math.abs(times[t] - o) < 0.05)).length
    return { times, flux, threshold, picked, onsets, tempo: (60 * RATE) / best, hits }
  }, [bpm.value, jitter.value, noise.value, seed.value])

  const top = Math.max(...r.flux)
  const series: XYSeries[] = [
    { name: 'onset envelope (spectral flux)', type: 'line', x: r.times, y: r.flux, slot: 0 },
    { name: 'adaptive threshold', type: 'line', x: r.times, y: r.threshold, dashed: true, slot: 2 },
    {
      name: 'detected onsets',
      type: 'scatter',
      x: r.picked.map((t) => r.times[t]),
      y: r.picked.map((t) => r.flux[t]),
      slot: 1,
    },
    { name: 'true onsets', type: 'scatter', x: r.onsets, y: r.onsets.map(() => top * 1.08), emphasis: true },
  ]

  return (
    <Interactive
      title="Onsets and tempo from spectral flux"
      caption="Four seconds of decaying notes, one per beat, with random pitches, timing jitter and noise. The onset envelope sums the increases in log-compressed magnitude between consecutive frames, so each note's attack is a sharp peak. Peaks above a moving median plus a margin are detected onsets; diamonds mark the true ones. The envelope's autocorrelation peaks at the beat period, which gives the tempo. Jitter and noise first add false and missed detections, and only later disturb the tempo estimate."
      controls={
        <>
          <ParamSlider label="tempo (BPM)" param={bpm} />
          <ParamSlider label="timing jitter (ms)" param={jitter} />
          <ParamSlider label="noise level" param={noise} />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="estimated tempo" value={`${formatNumber(r.tempo)} BPM`} />
          <Readout label="onsets found" value={`${r.hits} of ${r.onsets.length}`} />
          <Readout label="detections" value={r.picked.length} />
        </>
      }
    >
      <XYChart series={series} xLabel="time (s)" yLabel="flux" height={300} />
    </Interactive>
  )
}

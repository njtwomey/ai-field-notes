import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { stft } from '@/lib/dsp'
import { rng } from '@/lib/math'

const FS = 8000
const SIZE = 512
const HOP = 256
const BINS = 97 // 0–1.5 kHz
const LENGTH = 3 * FS

/** Two instruments: a low note repeating every 0.5 s and a higher note held in two long phrases. */
function mixture() {
  const low = new Array(LENGTH).fill(0)
  const high = new Array(LENGTH).fill(0)
  const note = (out: number[], f0: number, start: number, dur: number, decay: number) => {
    const n0 = Math.round(start * FS)
    for (let i = 0; i < dur * FS && n0 + i < LENGTH; i++) {
      const env = Math.exp(-i / (decay * FS))
      let s = 0
      for (let h = 1; h <= 5; h++) s += Math.cos((2 * Math.PI * h * f0 * i) / FS) / h
      out[n0 + i] += env * s
    }
  }
  for (let t = 0.1; t < 2.9; t += 0.5) note(low, 196, t, 0.3, 0.08)
  note(high, 523, 0.4, 0.9, 1)
  note(high, 523, 1.7, 1.0, 1)
  return { low, high, mix: low.map((v, n) => v + 0.6 * high[n]) }
}

const magnitudes = (x: number[]) => stft(x, SIZE, HOP).frames.map((f) => Array.from(f.slice(0, BINS)))

/**
 * NMF of the mixture's magnitude spectrogram V ≈ W H with the Kullback–Leibler multiplicative updates, then soft masks
 * W_k H_k / (W H) applied to V to separate each component.
 */
export function NmfSeparation() {
  const rank = useParam(2, { min: 1, max: 4, step: 1 })
  const iterations = useParam(120, { min: 1, max: 300, step: 1 })
  const data = useMemo(() => {
    return { V: magnitudes(mixture().mix) }
  }, [])

  const r = useMemo(() => {
    const V = data.V // frames × bins, i.e. V transposed; W is bins × K, H is K × frames
    const T = V.length
    const F = BINS
    const K = rank.value
    const g = rng(3)
    const W = Array.from({ length: F }, () => Array.from({ length: K }, () => 0.5 + g.uniform()))
    const H = Array.from({ length: K }, () => Array.from({ length: T }, () => 0.5 + g.uniform()))
    const approx = () =>
      Array.from({ length: F }, (_, f) =>
        Array.from({ length: T }, (_, t) => W[f].reduce((s, w, k) => s + w * H[k][t], 1e-12)),
      )
    for (let it = 0; it < iterations.value; it++) {
      let WH = approx()
      // H ← H ⊙ (Wᵀ (V / WH)) / (Wᵀ 1)
      for (let k = 0; k < K; k++) {
        const colSum = W.reduce((s, row) => s + row[k], 0)
        for (let t = 0; t < T; t++) {
          let num = 0
          for (let f = 0; f < F; f++) num += (W[f][k] * V[t][f]) / WH[f][t]
          H[k][t] *= num / colSum
        }
      }
      WH = approx()
      // W ← W ⊙ ((V / WH) Hᵀ) / (1 Hᵀ)
      for (let k = 0; k < K; k++) {
        const rowSum = H[k].reduce((s, v) => s + v, 0)
        for (let f = 0; f < F; f++) {
          let num = 0
          for (let t = 0; t < T; t++) num += (V[t][f] * H[k][t]) / WH[f][t]
          W[f][k] *= num / rowSum
        }
      }
    }
    const WH = approx()
    let kl = 0
    for (let f = 0; f < F; f++)
      for (let t = 0; t < T; t++) kl += V[t][f] * Math.log((V[t][f] + 1e-12) / WH[f][t]) - V[t][f] + WH[f][t]
    // The component whose activation varies most in time is the repeated low note.
    const busy = H.map((h) => {
      const m = h.reduce((a, b) => a + b, 0) / T
      return h.reduce((s, v) => s + (v - m) ** 2, 0)
    })
    const lowK = busy.indexOf(Math.max(...busy))
    const masked = Array.from({ length: F }, (_, f) =>
      Array.from({ length: T }, (_, t) => (V[t][f] * W[f][lowK] * H[lowK][t]) / WH[f][t]),
    )
    const dB = (grid: number[][]) => {
      const top = Math.max(...grid.flat())
      return grid.map((row) => row.map((v) => Math.max(-60, 20 * Math.log10(v / top + 1e-12))))
    }
    const mixture = Array.from({ length: F }, (_, f) => Array.from({ length: T }, (_, t) => V[t][f]))
    return { H, kl, mixture: dB(mixture), masked: dB(masked), lowK }
  }, [data, rank.value, iterations.value])

  const times = data.V.map((_, t) => (t * HOP + SIZE / 2) / FS)
  const freqs = Array.from({ length: BINS }, (_, k) => (k * FS) / SIZE)
  const activations: XYSeries[] = r.H.map((h, k) => ({
    name: `component ${k + 1}${k === r.lowK ? ' (masked below)' : ''}`,
    type: 'line',
    x: times,
    y: h,
    slot: k,
  }))

  return (
    <Interactive
      title="Separating two notes with NMF"
      caption="A repeated low note (196 Hz, every half second) and a sustained higher note (523 Hz) mixed together. NMF factorises the mixture's magnitude spectrogram into spectral templates W and activations H with the KL multiplicative updates. With rank 2, each component learns one note's harmonic template and its activations say when it sounds (middle). Masking the mixture with one component's share W_k H_k / W H recovers that note alone (bottom). Rank 1 cannot separate them; higher ranks split one note over several components."
      controls={
        <>
          <ParamSlider label="rank K" param={rank} withArrows />
          <ParamSlider label="iterations" param={iterations} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="generalised KL D(V ‖ WH)" value={formatNumber(r.kl)} />
        </>
      }
    >
      <div className="space-y-4">
        <Heatmap
          x={times}
          y={freqs}
          z={r.mixture}
          range={[-60, 0]}
          xLabel="time (s)"
          yLabel="frequency (Hz)"
          valueLabel="dB"
          height={220}
        />
        <XYChart series={activations} xLabel="time (s)" yLabel="activation" height={200} />
        <Heatmap
          x={times}
          y={freqs}
          z={r.masked}
          range={[-60, 0]}
          xLabel="time (s)"
          yLabel="frequency (Hz)"
          valueLabel="dB"
          height={220}
        />
      </div>
    </Interactive>
  )
}

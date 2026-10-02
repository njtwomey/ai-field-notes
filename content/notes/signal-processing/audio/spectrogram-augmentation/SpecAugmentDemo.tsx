import { useMemo } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, useParam } from 'aifn-render'
import { rng } from '@/lib/math'
import { applyBank, chirp, harmonicTone, melFilterBank, powerSpectrogram } from '../_shared/audio'

const FS = 16000
const LENGTH = 2 * FS
const SIZE = 400 // 25 ms
const HOP = 160 // 10 ms
const MELS = 80

/** Two seconds of speech-like test audio: a tone with a gliding pitch, a pause, a chirp, and a steady chord. */
function logMel(): number[][] {
  const tone = harmonicTone(180, FS, LENGTH, 15)
  const sweep = chirp(300, 3000, FS, LENGTH)
  const upper = harmonicTone(330, FS, LENGTH, 8)
  const chord = harmonicTone(262, FS, LENGTH, 8).map((v, n) => v + upper[n])
  const x = Array.from({ length: LENGTH }, (_, n) => {
    const t = n / FS
    if (t < 0.6) return tone[n]
    if (t < 0.75) return 0
    if (t < 1.3) return sweep[n]
    return chord[n]
  })
  const { power } = powerSpectrogram(x, SIZE, HOP, 512)
  const bank = melFilterBank(MELS, 512, FS, 20, FS / 2, 'slaney')
  const mel = applyBank(power, bank.filters).map((f) => f.map((v) => Math.log(v + 1e-8)))
  // Per-band normalisation to zero mean, as in the SpecAugment setup; masks then set values to 0 (the mean).
  const T = mel.length
  for (let m = 0; m < MELS; m++) {
    const mu = mel.reduce((s, f) => s + f[m], 0) / T
    mel.forEach((f) => (f[m] -= mu))
  }
  return mel // frames × bands
}

/** SpecAugment on a log-mel spectrogram: time warping, frequency masks and time masks, with their sampling rules. */
export function SpecAugmentDemo() {
  const W = useParam(40, { min: 0, max: 80, step: 1 })
  const F = useParam(27, { min: 0, max: 40, step: 1 })
  const mF = useParam(2, { min: 0, max: 4, step: 1 })
  const T = useParam(40, { min: 0, max: 100, step: 1 })
  const mT = useParam(2, { min: 0, max: 4, step: 1 })
  const seed = useParam(1, { min: 1, max: 40, step: 1 })
  const base = useMemo(() => logMel(), [])

  const r = useMemo(() => {
    const g = rng(seed.value)
    const tau = base.length
    const nu = MELS
    // Time warp: a centre point c in (W, τ − W) moves to c + w, w ~ U(−W, W); both sides stretch linearly.
    let warped = base
    if (W.value > 0 && tau > 2 * W.value) {
      const c = W.value + g.uniform() * (tau - 2 * W.value)
      const w = (2 * g.uniform() - 1) * W.value
      warped = Array.from({ length: tau }, (_, t) => {
        const src = t < c + w ? (t * c) / (c + w) : c + ((t - (c + w)) * (tau - c)) / (tau - c - w)
        const i = Math.min(tau - 2, Math.max(0, Math.floor(src)))
        const frac = Math.min(1, Math.max(0, src - i))
        return base[i].map((v, m) => (1 - frac) * v + frac * base[i + 1][m])
      })
    }
    const out = warped.map((f) => [...f])
    const freqMasks: [number, number][] = []
    for (let j = 0; j < mF.value; j++) {
      const f = Math.floor(g.uniform() * (F.value + 1))
      const f0 = Math.floor(g.uniform() * (nu - f))
      freqMasks.push([f0, f])
      out.forEach((frame) => frame.fill(0, f0, f0 + f))
    }
    const timeMasks: [number, number][] = []
    for (let j = 0; j < mT.value; j++) {
      const t = Math.floor(g.uniform() * (T.value + 1))
      const t0 = Math.floor(g.uniform() * (tau - t))
      timeMasks.push([t0, t])
      for (let i = t0; i < t0 + t; i++) out[i].fill(0)
    }
    const toRows = (grid: number[][]) => grid[0].map((_, m) => grid.map((f) => f[m]))
    return { original: toRows(base), augmented: toRows(out), freqMasks, timeMasks }
  }, [base, W.value, F.value, mF.value, T.value, mT.value, seed.value])

  const times = base.map((_, t) => (t * HOP) / FS)
  const bands = Array.from({ length: MELS }, (_, m) => m)
  const range: [number, number] = [-6, 6]

  return (
    <Interactive
      title="SpecAugment"
      caption="A normalised 80-band log-mel spectrogram (25 ms windows, 10 ms hop) before (top) and after (bottom) augmentation. Time warping moves a random centre point by up to W frames and stretches both sides; each frequency mask zeroes f ~ U[0, F] consecutive bands, and each time mask t ~ U[0, T] consecutive frames, at uniformly random positions. Zero is the mean after normalisation, so masked regions carry no information. Step the seed to draw new augmentations."
      controls={
        <>
          <ParamSlider label="time warp W (frames)" param={W} />
          <ParamSlider label="frequency mask F (bands)" param={F} />
          <ParamSlider label="frequency masks m_F" param={mF} />
          <ParamSlider label="time mask T (frames)" param={T} />
          <ParamSlider label="time masks m_T" param={mT} />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout
            label="frequency masks [start, width]"
            value={r.freqMasks.map(([a, b]) => `[${a}, ${b}]`).join(' ') || '—'}
          />
          <Readout
            label="time masks [start, width]"
            value={r.timeMasks.map(([a, b]) => `[${a}, ${b}]`).join(' ') || '—'}
          />
        </>
      }
    >
      <div className="space-y-4">
        <Heatmap
          x={times}
          y={bands}
          z={r.original}
          range={range}
          xLabel="time (s)"
          yLabel="mel band"
          valueLabel="normalised log energy"
          height={220}
        />
        <Heatmap
          x={times}
          y={bands}
          z={r.augmented}
          range={range}
          xLabel="time (s)"
          yLabel="mel band"
          valueLabel="normalised log energy"
          height={220}
        />
      </div>
    </Interactive>
  )
}

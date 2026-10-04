import { useMemo } from 'react'
import { Figure, int, Plot, Raster, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { applyBank, chirp, harmonicTone, melFilterBank, powerSpectrogram } from '../_shared/audio'
import { stream, uniform } from 'aifn/foundation/random'

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
  const state = useFigureState({
    W: slider(0, 80, 40, { step: 1, label: 'time warp W (frames)' }),
    F: int(27, { min: 0, max: 40, step: 1, label: 'frequency mask F (bands)' }),
    mF: int(2, { min: 0, max: 4, step: 1, label: 'frequency masks m_F' }),
    T: slider(0, 100, 40, { step: 1, label: 'time mask T (frames)' }),
    mT: slider(0, 4, 2, { step: 1, label: 'time masks m_T' }),
    seed: int(1, { min: 1, max: 40, step: 1, label: 'seed' }),
  })
  const base = useMemo(() => logMel(), [])

  const r = useMemo(() => {
    const g = stream(state.seed)
    const tau = base.length
    const nu = MELS
    // Time warp: a centre point c in (W, τ − W) moves to c + w, w ~ U(−W, W); both sides stretch linearly.
    let warped = base
    if (state.W > 0 && tau > 2 * state.W) {
      const c = state.W + uniform(g) * (tau - 2 * state.W)
      const w = (2 * uniform(g) - 1) * state.W
      warped = Array.from({ length: tau }, (_, t) => {
        const src = t < c + w ? (t * c) / (c + w) : c + ((t - (c + w)) * (tau - c)) / (tau - c - w)
        const i = Math.min(tau - 2, Math.max(0, Math.floor(src)))
        const frac = Math.min(1, Math.max(0, src - i))
        return base[i].map((v, m) => (1 - frac) * v + frac * base[i + 1][m])
      })
    }
    const out = warped.map((f) => [...f])
    const freqMasks: [number, number][] = []
    for (let j = 0; j < state.mF; j++) {
      const f = Math.floor(uniform(g) * (state.F + 1))
      const f0 = Math.floor(uniform(g) * (nu - f))
      freqMasks.push([f0, f])
      out.forEach((frame) => frame.fill(0, f0, f0 + f))
    }
    const timeMasks: [number, number][] = []
    for (let j = 0; j < state.mT; j++) {
      const t = Math.floor(uniform(g) * (state.T + 1))
      const t0 = Math.floor(uniform(g) * (tau - t))
      timeMasks.push([t0, t])
      for (let i = t0; i < t0 + t; i++) out[i].fill(0)
    }
    const toRows = (grid: number[][]) => grid[0].map((_, m) => grid.map((f) => f[m]))
    return { original: toRows(base), augmented: toRows(out), freqMasks, timeMasks }
  }, [base, state.W, state.F, state.mF, state.T, state.mT, state.seed])

  const times = base.map((_, t) => (t * HOP) / FS)
  const bands = Array.from({ length: MELS }, (_, m) => m)
  const range: [number, number] = [-6, 6]

  const xAxis = useAxis({ label: 'time (s)' })
  const yAxis = useAxis({ label: 'mel band' })
  const xAxis2 = useAxis({ label: 'time (s)' })
  const yAxis2 = useAxis({ label: 'mel band' })
  return (
    <Figure
      title="SpecAugment"
      state={state}
      caption="A normalised 80-band log-mel spectrogram (25 ms windows, 10 ms hop) before (top) and after (bottom) augmentation. Time warping moves a random centre point by up to W frames and stretches both sides; each frequency mask zeroes f ~ U[0, F] consecutive bands, and each time mask t ~ U[0, T] consecutive frames, at uniformly random positions. Zero is the mean after normalisation, so masked regions carry no information. Step the seed to draw new augmentations."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={220}>
          <Raster x={times} y={bands} z={r.original} range={range} valueLabel={'normalised log energy'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          <Raster x={times} y={bands} z={r.augmented} range={range} valueLabel={'normalised log energy'} />
        </Plot>
      </div>
    </Figure>
  )
}

import { useMemo } from 'react'
import { choice, Figure, float, formatNumber, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { stft } from '../_shared/audio'
import { normal, stream } from 'aifn/foundation/random'

const FS = 8000
const LENGTH = 2 * FS
const SIZE = 256
const HOP = 128
const LEAD = 0.3 // seconds of noise before the tone, used to estimate the noise spectrum

type Method = 'subtraction' | 'wiener'

/**
 * Magnitude-domain speech enhancement on a gliding harmonic tone in white noise. The noise power spectrum is averaged
 * over the leading noise-only frames. Spectral subtraction removes α times that power with a spectral floor β; the
 * Wiener rule applies SNR/(1+SNR) with the decision-directed a priori SNR.
 */
export function SpectralSubtraction() {
  const state = useFigureState({
    method: choice<Method>(
      [
        { value: 'subtraction', label: 'power subtraction' },
        { value: 'wiener', label: 'Wiener, decision-directed' },
      ],
      'subtraction',
      { label: 'method' },
    ),
    snrDb: int(0, { min: -10, max: 15, step: 1, label: 'input SNR (dB)' }),
    alpha: float(2, { min: 1, max: 5, step: 0.1, label: 'over-subtraction α' }),
    floor: float(0.02, { min: 0, max: 0.3, step: 0.01, label: 'spectral floor β' }),
  })

  const r = useMemo(() => {
    const g = stream(8)
    // A tone gliding from 200 to 300 Hz with 8 harmonics, silent for the first LEAD seconds.
    const clean = Array.from({ length: LENGTH }, (_, n) => {
      const t = n / FS
      if (t < LEAD) return 0
      const u = t - LEAD
      const phase = 2 * Math.PI * (200 * u + (100 * u * u) / (2 * (LENGTH / FS - LEAD)))
      let s = 0
      for (let h = 1; h <= 8; h++) s += Math.cos(h * phase) / h
      return s
    })
    const cleanPower = clean.reduce((a, v) => a + v * v, 0) / (LENGTH * (1 - LEAD / (LENGTH / FS)))
    const sigma = Math.sqrt(cleanPower / 10 ** (state.snrDb / 10))
    const noisy = clean.map((v) => v + sigma * normal(g))
    const X = stft(noisy, SIZE, HOP).frames
    const S = stft(clean, SIZE, HOP).frames
    const leadFrames = Math.floor((LEAD * FS - SIZE) / HOP) + 1
    const bins = X[0].length
    const noisePower = Array.from(
      { length: bins },
      (_, k) => X.slice(0, leadFrames).reduce((a, f) => a + f[k] * f[k], 0) / leadFrames,
    )
    const enhanced: number[][] = []
    for (const frame of X) {
      const prevClean = enhanced.at(-1) ?? new Array<number>(bins).fill(0)
      const out = Array.from(frame, (m, k) => {
        const p = m * m
        if (state.method === 'subtraction') return Math.sqrt(Math.max(p - state.alpha * noisePower[k], state.floor * p))
        // Decision-directed a priori SNR (Ephraim–Malah), then the Wiener gain.
        const post = p / noisePower[k]
        const prio = 0.98 * ((prevClean[k] * prevClean[k]) / noisePower[k]) + 0.02 * Math.max(post - 1, 0)
        return m * (prio / (1 + prio))
      })
      enhanced.push(out)
    }
    const snr = (est: number[][]) => {
      let sig = 0
      let err = 0
      S.forEach((f, t) => f.forEach((v, k) => ((sig += v * v), (err += (est[t][k] - v) ** 2))))
      return 10 * Math.log10(sig / err)
    }
    const toDb = (grid: number[][]) => {
      const top = Math.max(...grid.flat())
      return grid[0].map((_, k) => grid.map((f) => Math.max(-60, 20 * Math.log10(f[k] / top + 1e-12))))
    }
    const Xarr = X.map((f) => Array.from(f))
    return {
      times: X.map((_, t) => (t * HOP + SIZE / 2) / FS),
      freqs: Array.from({ length: bins }, (_, k) => (k * FS) / SIZE),
      noisyDb: toDb(Xarr),
      enhancedDb: toDb(enhanced),
      before: snr(Xarr),
      after: snr(enhanced),
    }
  }, [state.method, state.snrDb, state.alpha, state.floor])

  const xAxis = useAxis({ label: 'time (s)' })
  const yAxis = useAxis({ label: 'frequency (Hz)' })
  const xAxis2 = useAxis({ label: 'time (s)' })
  const yAxis2 = useAxis({ label: 'frequency (Hz)' })
  return (
    <Figure
      title="Spectral subtraction and the Wiener rule"
      state={state}
      caption="A gliding harmonic tone in white noise; the first 0.3 s is noise only and gives the noise estimate. Top: the noisy spectrogram. Bottom: after enhancement. Plain subtraction (α = 1, no floor) leaves isolated random peaks wherever the noise happened to exceed its average: musical noise. Over-subtraction and a spectral floor suppress them at the cost of thinning the signal. The Wiener rule with a decision-directed a priori SNR smooths the gain over time and gives much less musical noise. The readouts compare magnitudes with the clean signal's."

      readouts={
        <>
          <Readout label="magnitude SNR, noisy" value={`${formatNumber(r.before)} dB`} />
          <Readout label="magnitude SNR, enhanced" value={`${formatNumber(r.after)} dB`} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={220}>
          <Raster x={r.times} y={r.freqs} z={r.noisyDb} range={[-60, 0]} valueLabel={'dB'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          <Raster x={r.times} y={r.freqs} z={r.enhancedDb} range={[-60, 0]} valueLabel={'dB'} />
        </Plot>
      </div>
    </Figure>
  )
}

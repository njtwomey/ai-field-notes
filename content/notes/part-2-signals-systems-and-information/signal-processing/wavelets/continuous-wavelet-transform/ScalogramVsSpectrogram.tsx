import { useMemo } from 'react'
import { Figure, formatNumber, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { rfft } from 'aifn-compute/foundation/fourier'
import { complexAbs, toFlat } from 'aifn-compute/foundation/tensor'
import { getWindow } from 'aifn-compute/signal'
import { morletCwt } from '../_shared/wavelets'

/** Decibels, 20 log₁₀ of a magnitude, floored so zeros stay finite. */
const db = (magnitude: number, floor: number) => Math.max(floor, 20 * Math.log10(Math.max(magnitude, 1e-300)))

/**
 * Short-time Fourier transform magnitudes: Hann-windowed frames of `size` samples every `hop` samples, padded to `nfft`.
 * Returns frames × (nfft/2 + 1) magnitudes and the frame centres in samples.
 */
function stft(x: ArrayLike<number>, size: number, hop: number, nfft: number) {
  const w = toFlat(getWindow('hann', size, { periodic: true }))
  const frames: number[][] = []
  const centres: number[] = []
  for (let start = 0; start + size <= x.length; start += hop) {
    const frame = Float64Array.from(w, (wi, i) => x[start + i] * wi)
    frames.push(toFlat(complexAbs(rfft(frame, { n: nfft }))))
    centres.push(start + size / 2)
  }
  return { frames, centres }
}

const FS = 1000
const N = 1024
const FLOOR = 40
const COLUMN_STEP = 4
// 64 frequencies spaced evenly in log₂ f from 5 Hz to 400 Hz.
const LOG2_F = Array.from({ length: 64 }, (_, i) => Math.log2(5) + ((Math.log2(400) - Math.log2(5)) * i) / 63)
const FREQS = LOG2_F.map((v) => 2 ** v)

// A 20 Hz tone throughout, a 250 Hz burst 12 ms wide at 0.3 s, a click at 0.7 s, and a 90 Hz tone after 0.5 s.
const SIGNAL = (() => {
  const x = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    const t = n / FS
    x[n] = Math.cos(2 * Math.PI * 20 * t)
    x[n] += 1.5 * Math.exp(-(((t - 0.3) / 0.006) ** 2)) * Math.cos(2 * Math.PI * 250 * t)
    if (t > 0.5) x[n] += 0.8 * Math.cos(2 * Math.PI * 90 * t)
  }
  x[Math.round(0.7 * FS)] += 6
  return x
})()

/**
 * A scalogram uses a window that shrinks as frequency rises, so it resolves low tones finely in frequency and high-
 * frequency events finely in time. A spectrogram uses one window for every frequency.
 */
export function ScalogramVsSpectrogram() {
  const state = useFigureState({
    omega0: int(6, { min: 4, max: 16, step: 1, label: 'Morlet ω₀', format: (v) => String(v) }),
    exponent: int(7, { min: 5, max: 9, step: 1, label: 'spectrogram window (samples)', format: (v) => String(2 ** v) }),
  })
  const size = 2 ** state.exponent

  const cwt = useMemo(() => {
    const rows = morletCwt(SIGNAL, FS, FREQS, state.omega0)
    const cols = Array.from({ length: N / COLUMN_STEP }, (_, i) => i * COLUMN_STEP)
    let peak = 0
    for (const row of rows) for (const i of cols) peak = Math.max(peak, row[i])
    return {
      times: cols.map((i) => i / FS),
      z: rows.map((row) => cols.map((i) => db(row[i] / peak, -FLOOR))),
    }
  }, [state.omega0])

  const spec = useMemo(() => {
    const { frames, centres } = stft(SIGNAL, size, Math.max(2, size / 8), Math.max(size, 256))
    const nfft = Math.max(size, 256)
    const bins = Array.from({ length: nfft / 2 + 1 }, (_, k) => (k * FS) / nfft).filter((f) => f <= 400)
    const peak = Math.max(...frames.map((f) => Math.max(...f)))
    return {
      times: centres.map((c) => c / FS),
      freqs: bins,
      z: bins.map((_, k) => frames.map((f) => db(f[k] / peak, -FLOOR))),
    }
  }, [size])

  // Morlet in time: the Gaussian envelope has σ = a seconds with a = ω₀/(2πf); in frequency, σ = 1/(2πa) Hz.
  const sigmaT100 = (1000 * state.omega0) / (2 * Math.PI * 100)
  const sigmaF100 = 100 / state.omega0

  const xAxis = useAxis({ label: 'time (s)' })
  const yAxis = useAxis({ label: 'log₂ frequency (Hz)' })
  const xAxis2 = useAxis({ label: 'time (s)' })
  const yAxis2 = useAxis({ label: 'frequency (Hz)' })
  return (
    <Figure
      title="Scalogram against spectrogram"
      state={state}
      caption="A 20 Hz tone throughout, a 250 Hz burst 12 ms wide at 0.3 s, a click at 0.7 s, and a 90 Hz tone from 0.5 s, sampled at 1 kHz. Left: the Morlet scalogram on a logarithmic frequency axis. Right: a Hann spectrogram on a linear axis, up to 400 Hz. The wavelet's time window shrinks as frequency rises, so the burst and click stay sharp in time while the 20 Hz tone stays sharp in frequency. The spectrogram's single window must choose one. Raising ω₀ lengthens every wavelet and moves the scalogram toward sharper frequency and blurrier time."

      readouts={
        <>
          <Readout label="wavelet σ_t at 100 Hz" value={`${formatNumber(sigmaT100)} ms`} />
          <Readout label="wavelet σ_f at 100 Hz" value={`${formatNumber(sigmaF100)} Hz`} />
          <Readout label="spectrogram window" value={`${formatNumber((1000 * size) / FS)} ms`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">Morlet scalogram (dB)</div>
          <Plot x={xAxis} y={yAxis} height={340}>
            <Raster x={cwt.times} y={LOG2_F} z={cwt.z} range={[-FLOOR, 0]} valueLabel={'dB'} />
          </Plot>
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">Hann spectrogram (dB)</div>
          <Plot x={xAxis2} y={yAxis2} height={340}>
            <Raster x={spec.times} y={spec.freqs} z={spec.z} range={[-FLOOR, 0]} valueLabel={'dB'} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}

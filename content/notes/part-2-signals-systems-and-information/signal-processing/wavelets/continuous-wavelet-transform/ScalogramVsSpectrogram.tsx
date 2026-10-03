import { useMemo } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, formatNumber, useParam } from 'aifn-render'
import { db, stft } from '@/lib/dsp'
import { morletCwt } from '../_shared/wavelets'

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
  const omega0 = useParam(6, { min: 4, max: 16, step: 1 })
  const exponent = useParam(7, { min: 5, max: 9, step: 1 })
  const size = 2 ** exponent.value

  const cwt = useMemo(() => {
    const rows = morletCwt(SIGNAL, FS, FREQS, omega0.value)
    const cols = Array.from({ length: N / COLUMN_STEP }, (_, i) => i * COLUMN_STEP)
    let peak = 0
    for (const row of rows) for (const i of cols) peak = Math.max(peak, row[i])
    return {
      times: cols.map((i) => i / FS),
      z: rows.map((row) => cols.map((i) => db(row[i] / peak, -FLOOR))),
    }
  }, [omega0.value])

  const spec = useMemo(() => {
    const { frames, centres } = stft(SIGNAL, size, Math.max(2, size / 8), 'hann', Math.max(size, 256))
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
  const sigmaT100 = (1000 * omega0.value) / (2 * Math.PI * 100)
  const sigmaF100 = 100 / omega0.value

  return (
    <Interactive
      title="Scalogram against spectrogram"
      caption="A 20 Hz tone throughout, a 250 Hz burst 12 ms wide at 0.3 s, a click at 0.7 s, and a 90 Hz tone from 0.5 s, sampled at 1 kHz. Left: the Morlet scalogram on a logarithmic frequency axis. Right: a Hann spectrogram on a linear axis, up to 400 Hz. The wavelet's time window shrinks as frequency rises, so the burst and click stay sharp in time while the 20 Hz tone stays sharp in frequency. The spectrogram's single window must choose one. Raising ω₀ lengthens every wavelet and moves the scalogram toward sharper frequency and blurrier time."
      controls={
        <>
          <ParamSlider label="Morlet ω₀" param={omega0} format={(v) => String(v)} withArrows />
          <ParamSlider
            label="spectrogram window (samples)"
            param={exponent}
            format={(v) => String(2 ** v)}
            withArrows
          />
        </>
      }
      readout={
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
          <Heatmap
            x={cwt.times}
            y={LOG2_F}
            z={cwt.z}
            range={[-FLOOR, 0]}
            xLabel="time (s)"
            yLabel="log₂ frequency (Hz)"
            valueLabel="dB"
            height={340}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">Hann spectrogram (dB)</div>
          <Heatmap
            x={spec.times}
            y={spec.freqs}
            z={spec.z}
            range={[-FLOOR, 0]}
            xLabel="time (s)"
            yLabel="frequency (Hz)"
            valueLabel="dB"
            height={340}
          />
        </div>
      </div>
    </Interactive>
  )
}

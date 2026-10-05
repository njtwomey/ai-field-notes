import { useMemo } from 'react'
import { choice, Figure, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { applyBank, harmonicTone, melFilterBank, powerSpectrogram } from '../_shared/audio'
import { normal, stream } from 'aifn-compute/foundation/random'

const FS = 16000
const LENGTH = FS // one second

/** One second of test audio: a harmonic tone, an upward chirp, a noise burst, then a two-note chord. */
function testSignal(): number[] {
  const g = stream(5)
  const tone = harmonicTone(220, FS, LENGTH, 12)
  const upper = harmonicTone(415, FS, LENGTH, 8)
  const chord = harmonicTone(330, FS, LENGTH, 8).map((v, n) => v + upper[n])
  return Array.from({ length: LENGTH }, (_, n) => {
    const t = n / FS
    if (t < 0.3) return 0.6 * tone[n]
    if (t < 0.6) {
      const u = t - 0.3
      return 0.8 * Math.cos(2 * Math.PI * (300 * u + ((4000 - 300) * u * u) / (2 * 0.3)))
    }
    if (t < 0.68) return 0.5 * normal(g)
    if (t < 0.72) return 0
    return 0.4 * chord[n]
  })
}

type Size = '256' | '512' | '1024'

/**
 * The same signal as a linear-frequency power spectrogram and as a log-mel spectrogram. Both in dB, floored 80 dB
 * below the maximum.
 */
export function MelSpectrogramView() {
  const state = useFigureState({
    size: choice<Size>(
      [
        { value: '256', label: '256 (16 ms)' },
        { value: '512', label: '512 (32 ms)' },
        { value: '1024', label: '1024 (64 ms)' },
      ],
      '512',
      { label: 'window length (samples)' },
    ),
    mels: int(64, { min: 16, max: 128, step: 8, label: 'mel bands' }),
  })
  const signal = useMemo(() => testSignal(), [])

  const r = useMemo(() => {
    const n = Number(state.size)
    const hop = n / 4
    const { power, centres } = powerSpectrogram(signal, n, hop)
    const bank = melFilterBank(state.mels, n, FS, 0, FS / 2, 'slaney')
    const melPower = applyBank(power, bank.filters)
    const toDb = (grid: number[][]) => {
      const flat = grid.flat()
      const top = 10 * Math.log10(Math.max(...flat))
      // Rows are frequency bins and columns frames, for the heatmap: z[row][frame].
      return grid[0].map((_, k) => grid.map((frame) => Math.max(top - 80, 10 * Math.log10(frame[k] + 1e-20)) - top))
    }
    const times = centres.map((c) => c / FS)
    const bins = power[0].map((_, k) => (k * FS) / n)
    return {
      times,
      bins,
      linear: toDb(power),
      mel: toDb(melPower),
      melIndex: bank.filters.map((_, m) => m),
      frames: power.length,
      hop,
    }
  }, [signal, state.size, state.mels])

  const xAxis = useAxis({ label: 'time (s)' })
  const yAxis = useAxis({ label: 'frequency (Hz)' })
  const xAxis2 = useAxis({ label: 'time (s)' })
  const yAxis2 = useAxis({ label: 'mel band' })
  return (
    <Figure
      title="Linear and mel spectrograms"
      state={state}
      caption="One second of test audio: a 220 Hz harmonic tone, a chirp from 300 to 4000 Hz, a noise burst, and a two-note chord. Top: the power spectrogram on a linear frequency axis, with a Hann window and a hop of a quarter window. Bottom: the log-mel spectrogram, the same power summed through a mel filter bank, which spends most of its rows below 2 kHz. Longer windows sharpen the harmonics but smear the chirp and the burst in time."

      readouts={
        <>
          <Readout label="frames" value={r.frames} />
          <Readout label="hop" value={`${r.hop} samples`} />
          <Readout label="frequency bins" value={r.bins.length} />
          <Readout label="mel bands" value={state.mels} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Raster x={r.times} y={r.bins} z={r.linear} range={[-80, 0]} valueLabel={'dB'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          <Raster x={r.times} y={r.melIndex} z={r.mel} range={[-80, 0]} valueLabel={'dB'} />
        </Plot>
      </div>
    </Figure>
  )
}

import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSlider, Readout, useParam } from 'aifn-render'
import { rng } from '@/lib/math'
import { applyBank, harmonicTone, melFilterBank, powerSpectrogram } from '../_shared/audio'

const FS = 16000
const LENGTH = FS // one second

/** One second of test audio: a harmonic tone, an upward chirp, a noise burst, then a two-note chord. */
function testSignal(): number[] {
  const g = rng(5)
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
    if (t < 0.68) return 0.5 * g.normal()
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
  const [size, setSize] = useState<Size>('512')
  const mels = useParam(64, { min: 16, max: 128, step: 8 })
  const signal = useMemo(() => testSignal(), [])

  const r = useMemo(() => {
    const n = Number(size)
    const hop = n / 4
    const { power, centres } = powerSpectrogram(signal, n, hop)
    const bank = melFilterBank(mels.value, n, FS, 0, FS / 2, 'slaney')
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
  }, [signal, size, mels.value])

  return (
    <Interactive
      title="Linear and mel spectrograms"
      caption="One second of test audio: a 220 Hz harmonic tone, a chirp from 300 to 4000 Hz, a noise burst, and a two-note chord. Top: the power spectrogram on a linear frequency axis, with a Hann window and a hop of a quarter window. Bottom: the log-mel spectrogram, the same power summed through a mel filter bank, which spends most of its rows below 2 kHz. Longer windows sharpen the harmonics but smear the chirp and the burst in time."
      controls={
        <>
          <ParamChoice
            label="window length (samples)"
            value={size}
            onChange={setSize}
            options={[
              { value: '256', label: '256 (16 ms)' },
              { value: '512', label: '512 (32 ms)' },
              { value: '1024', label: '1024 (64 ms)' },
            ]}
          />
          <ParamSlider label="mel bands" param={mels} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="frames" value={r.frames} />
          <Readout label="hop" value={`${r.hop} samples`} />
          <Readout label="frequency bins" value={r.bins.length} />
          <Readout label="mel bands" value={mels.value} />
        </>
      }
    >
      <div className="space-y-4">
        <Heatmap
          x={r.times}
          y={r.bins}
          z={r.linear}
          range={[-80, 0]}
          xLabel="time (s)"
          yLabel="frequency (Hz)"
          valueLabel="dB"
          height={260}
        />
        <Heatmap
          x={r.times}
          y={r.melIndex}
          z={r.mel}
          range={[-80, 0]}
          xLabel="time (s)"
          yLabel="mel band"
          valueLabel="dB"
          height={260}
        />
      </div>
    </Interactive>
  )
}

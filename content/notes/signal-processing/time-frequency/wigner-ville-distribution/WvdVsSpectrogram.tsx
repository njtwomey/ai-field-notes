import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, Readout, formatNumber } from '@/components/viz'
import { db, stft } from '@/lib/dsp'
import { analytic, chirp, wignerVille } from '../_shared/tf'

type Choice = 'chirp' | 'tones' | 'both'

const N = 128
const WINDOW = 32
const FLOOR = 50

/** Test signals in cycles per sample (f_s = 1), all below 0.5. */
function makeSignal(choice: Choice): Float64Array {
  if (choice === 'chirp') return chirp(N, 1, 0.05, 0.42)
  const x = new Float64Array(N)
  if (choice === 'tones') {
    for (let n = 0; n < N; n++) x[n] = Math.cos(2 * Math.PI * 0.12 * n) + Math.cos(2 * Math.PI * 0.32 * n)
    return x
  }
  const c = chirp(N, 1, 0.05, 0.25)
  for (let n = 0; n < N; n++) x[n] = c[n] + Math.cos(2 * Math.PI * 0.38 * n)
  return x
}

/**
 * The Wigner–Ville distribution concentrates a linear chirp on its instantaneous-frequency line, where a spectrogram
 * smears it over the window's bandwidth. With two components it adds oscillating cross-terms midway between them.
 */
export function WvdVsSpectrogram() {
  const [choice, setChoice] = useState<Choice>('chirp')

  const r = useMemo(() => {
    const x = makeSignal(choice)
    // Wigner–Ville of the analytic signal: rows are time, columns frequency k/(2N) cycles per sample.
    const wvd = wignerVille(analytic(x))
    const maxAbs = Math.max(...wvd.map((row) => Math.max(...row.map(Math.abs))))
    const freqs = Array.from({ length: N }, (_, k) => k / (2 * N))
    const times = Array.from({ length: N }, (_, n) => n)
    const zW = freqs.map((_, k) => times.map((n) => wvd[n][k] / maxAbs))
    let minW = 0
    for (const row of zW) for (const v of row) minW = Math.min(minW, v)
    // Spectrogram with a 32-sample Hann window, one frame every 2 samples, zero-padded to 256 bins over [0, 0.5].
    const { frames, centres } = stft(x, WINDOW, 2, 'hann', 256)
    const peak = Math.max(...frames.map((f) => Math.max(...f)))
    const sFreqs = Array.from({ length: frames[0].length }, (_, k) => k / 256)
    const zS = sFreqs.map((_, k) => frames.map((f) => db(f[k] / peak, -FLOOR)))
    return { zW, freqs, times, zS, sFreqs, sTimes: centres, minW }
  }, [choice])

  return (
    <Interactive
      title="Sharper than a spectrogram, with cross-terms"
      caption="Left: the spectrogram with a 32-sample Hann window, in dB. Right: the Wigner–Ville distribution of the analytic signal, on a diverging scale because it takes negative values. For a single linear chirp the Wigner–Ville distribution lies on the chirp's instantaneous frequency, while the spectrogram spreads it over the window's bandwidth. With two tones it adds a third, oscillating ridge midway between them that belongs to neither: an interference term, positive and negative in alternation."
      controls={
        <ParamChoice
          label="signal"
          value={choice}
          onChange={setChoice}
          options={[
            { value: 'chirp', label: 'linear chirp' },
            { value: 'tones', label: 'two tones' },
            { value: 'both', label: 'chirp + tone' },
          ]}
        />
      }
      readout={<Readout label="most negative Wigner–Ville value (relative)" value={formatNumber(r.minW)} />}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">spectrogram (dB)</div>
          <Heatmap
            x={r.sTimes}
            y={r.sFreqs}
            z={r.zS}
            range={[-FLOOR, 0]}
            xLabel="time (samples)"
            yLabel="frequency (cycles/sample)"
            valueLabel="dB"
            height={320}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">Wigner–Ville distribution</div>
          <Heatmap
            x={r.times}
            y={r.freqs}
            z={r.zW}
            scale="diverging"
            range={[-1, 1]}
            xLabel="time (samples)"
            yLabel="frequency (cycles/sample)"
            valueLabel="W / max|W|"
            height={320}
          />
        </div>
      </div>
    </Interactive>
  )
}

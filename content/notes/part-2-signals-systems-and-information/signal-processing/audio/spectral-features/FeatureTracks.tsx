import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'
import { chirp, harmonicTone, powerSpectrogram } from '../_shared/audio'

const FS = 16000
const LENGTH = 16000
const SIZE = 512
const HOP = 256

/** A harmonic tone, white noise, then a rising chirp: three sounds with very different spectral shapes. */
function testSignal(noiseLevel: number) {
  const g = rng(9)
  const tone = harmonicTone(220, FS, LENGTH, 10)
  const sweep = chirp(200, 6000, FS, LENGTH)
  return Array.from({ length: LENGTH }, (_, n) => {
    const t = n / FS
    const base = t < 0.33 ? 0.5 * tone[n] : t < 0.66 ? 0.5 * g.normal() : 0.7 * sweep[n]
    return base + noiseLevel * g.normal()
  })
}

/**
 * Frame-level spectral features over time: centroid and 85% roll-off in Hz, and flatness and zero-crossing rate on
 * a 0–1 scale.
 */
export function FeatureTracks() {
  const noise = useParam(0, { min: 0, max: 0.3, step: 0.01 })

  const r = useMemo(() => {
    const x = testSignal(noise.value)
    const { power, centres } = powerSpectrogram(x, SIZE, HOP)
    const freqs = power[0].map((_, k) => (k * FS) / SIZE)
    const features = power.map((P, t) => {
      const total = P.reduce((a, b) => a + b, 0) + 1e-20
      const centroid = P.reduce((a, p, k) => a + p * freqs[k], 0) / total
      let cum = 0
      let rolloff = freqs[freqs.length - 1]
      for (let k = 0; k < P.length; k++) {
        cum += P[k]
        if (cum >= 0.85 * total) {
          rolloff = freqs[k]
          break
        }
      }
      const logMean = P.reduce((a, p) => a + Math.log(p + 1e-20), 0) / P.length
      const flatness = Math.exp(logMean) / (total / P.length)
      const start = t * HOP
      let crossings = 0
      for (let n = start + 1; n < start + SIZE; n++) if (Math.sign(x[n]) !== Math.sign(x[n - 1])) crossings++
      return { centroid, rolloff, flatness, zcr: crossings / (SIZE - 1) }
    })
    return { times: centres.map((c) => c / FS), features }
  }, [noise.value])

  const hz: XYSeries[] = [
    { name: 'spectral centroid', type: 'line', x: r.times, y: r.features.map((f) => f.centroid), slot: 0 },
    { name: '85% roll-off', type: 'line', x: r.times, y: r.features.map((f) => f.rolloff), slot: 1 },
  ]
  const unit: XYSeries[] = [
    { name: 'spectral flatness', type: 'line', x: r.times, y: r.features.map((f) => f.flatness), slot: 2 },
    { name: 'zero-crossing rate (per sample)', type: 'line', x: r.times, y: r.features.map((f) => f.zcr), slot: 3 },
  ]
  const mid = r.features[Math.floor(r.features.length / 2)]

  return (
    <Interactive
      title="Spectral features over time"
      caption="Three sounds in turn: a 220 Hz harmonic tone, white noise, and a chirp rising from 200 Hz to 6 kHz, with optional added noise. Left: the centroid and 85% roll-off track where the energy sits, low for the tone, near the middle of the band for noise, and rising with the chirp. Right: flatness is near 0 for tonal sounds and about 0.56 for white noise, e^{−γ}, because white-noise periodogram bins are exponentially distributed; the zero-crossing rate follows the dominant frequency, about 2f/f_s per sample for a pure tone. Adding noise pulls every feature towards the noise values."
      controls={<ParamSlider label="added noise level" param={noise} />}
      readout={
        <>
          <Readout label="noise section centroid" value={`${formatNumber(mid.centroid)} Hz`} />
          <Readout label="noise section flatness" value={formatNumber(mid.flatness)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart series={hz} xLabel="time (s)" yLabel="Hz" yRange={[0, FS / 2]} height={280} />
        <XYChart series={unit} xLabel="time (s)" yLabel="value" yRange={[0, 1]} height={280} />
      </div>
    </Interactive>
  )
}

import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { binFrequencies, db, freqz, lfilter, magnitudeSpectrum, makeWindow } from '@/lib/dsp'
import { autocorrelation, levinsonDurbin } from '../_shared/stat'

const FS = 8000
const N = 512
const FORMANTS: [number, number][] = [
  [700, 130],
  [1220, 70],
  [2600, 160],
]

/** Denominator A(z) of an all-pole filter with the given formant frequencies and bandwidths (Hz). */
function formantFilter(): number[] {
  let a = [1]
  for (const [f, b] of FORMANTS) {
    const r = Math.exp((-Math.PI * b) / FS)
    const theta = (2 * Math.PI * f) / FS
    const section = [1, -2 * r * Math.cos(theta), r * r]
    const next = new Array(a.length + 2).fill(0)
    a.forEach((ai, i) => section.forEach((sj, j) => (next[i + j] += ai * sj)))
    a = next
  }
  return a
}

/**
 * A vowel-like frame (an impulse train through three formant resonances) and its LPC spectral envelope
 * √E / |A(e^{iω})| from the Levinson–Durbin recursion, over the frame's windowed magnitude spectrum.
 */
export function LpcEnvelope() {
  const order = useParam(10, { min: 1, max: 24, step: 1 })
  const f0 = useParam(120, { min: 80, max: 300, step: 5 })

  const r = useMemo(() => {
    const a = formantFilter()
    const excitation = Array.from({ length: 3000 }, (_, n) => (n % Math.round(FS / f0.value) === 0 ? 1 : 0))
    const voiced = lfilter([1], a, excitation).slice(1000, 1000 + N)
    const w = makeWindow('hamming', N)
    const frame = voiced.map((v, n) => v * w[n])
    const rx = autocorrelation(frame, order.value)
    const { a: coeffs, reflection, error } = levinsonDurbin(rx, order.value)
    // Prediction-error filter A(z) = 1 − Σ a_k z^{−k}; envelope gain chosen to match the frame's power.
    const errorFilter = [1, ...coeffs.map((c) => -c)]
    const nfft = 1024
    const spectrum = magnitudeSpectrum(frame, nfft)
    const freqs = binFrequencies(nfft, FS)
    const env = freqz([Math.sqrt(error * N)], errorFilter, nfft / 2 + 1)
    const truth = freqz([Math.sqrt(error * N)], a, nfft / 2 + 1)
    return {
      freqs,
      spectrum: Array.from(spectrum, (v) => db(v)),
      envelope: env.magnitude.map((v) => db(v)),
      truth: truth.magnitude.map((v) => db(v)),
      gain: 10 * Math.log10(rx[0] / error),
      stable: reflection.every((k) => Math.abs(k) < 1),
    }
  }, [order.value, f0.value])

  const series: XYSeries[] = [
    { name: 'frame spectrum |X|', type: 'line', x: r.freqs, y: r.spectrum, muted: true },
    { name: 'true formant envelope', type: 'line', x: r.freqs, y: r.truth, slot: 2, dashed: true },
    { name: `LPC envelope, order ${order.value}`, type: 'line', x: r.freqs, y: r.envelope, slot: 1 },
  ]
  const top = Math.max(...r.spectrum)

  return (
    <Interactive
      title="The LPC spectral envelope"
      caption="A vowel-like frame: a pulse train at f₀ through resonances at 700, 1220 and 2600 Hz, Hamming-windowed. The order-p predictor from the Levinson–Durbin recursion defines an all-pole envelope √E/|A(e^{iω})| that traces the formants and ignores the harmonics. Three resonances need order 6; the prediction gain stops improving there. Very high orders start to fit individual harmonics, especially at high f₀."
      controls={
        <>
          <ParamSlider label="predictor order p" param={order} withArrows />
          <ParamSlider label="fundamental f₀ (Hz)" param={f0} />
        </>
      }
      readout={
        <>
          <Readout label="prediction gain r[0]/E" value={`${formatNumber(r.gain)} dB`} />
          <Readout label="all |kₘ| < 1 (stable)" value={r.stable ? 'yes' : 'no'} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="frequency (Hz)"
        yLabel="magnitude (dB)"
        xRange={[0, FS / 2]}
        yRange={[top - 70, top + 10]}
        height={340}
      />
    </Interactive>
  )
}

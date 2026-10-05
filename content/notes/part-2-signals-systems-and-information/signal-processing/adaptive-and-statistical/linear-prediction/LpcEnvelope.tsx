import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { rfft } from 'aifn-compute/foundation/fourier'
import { complexAbs, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { freqz, lfilter } from 'aifn-compute/signal/filters'
import { getWindow } from 'aifn-compute/signal/windows'
import { transferFunction } from 'aifn-compute/systems'
import { autocorrelation, levinsonDurbin } from '../_shared/stat'

const FS = 8000

/** Decibels, 20 log₁₀ of a magnitude, floored at −200 dB so zeros stay finite. */
const db = (m: number) => Math.max(-200, 20 * Math.log10(Math.max(m, 1e-300)))

/** |H(e^{iω})| of B(z)/A(z) at `count` frequencies ω from 0 to π inclusive. */
const responseMagnitude = (b: number[], a: number[], count: number) =>
  toFlat(complexAbs(freqz(transferFunction(b, a, { dt: 1 }), { n: count, includeNyquist: true }).values) as Tensor)
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
  const state = useFigureState({
    order: int(10, { min: 1, max: 24, step: 1, label: 'predictor order p' }),
    f0: int(120, { min: 80, max: 300, step: 5, label: 'fundamental f₀ (Hz)' }),
  })

  const r = useMemo(() => {
    const a = formantFilter()
    const excitation = Array.from({ length: 3000 }, (_, n) => (n % Math.round(FS / state.f0) === 0 ? 1 : 0))
    const voiced = toFlat(lfilter({ b: [1], a }, excitation).y as Tensor).slice(1000, 1000 + N)
    const w = toFlat(getWindow('hamming', N))
    const frame = voiced.map((v, n) => v * w[n])
    const rx = autocorrelation(frame, state.order)
    const { a: coeffs, reflection, error } = levinsonDurbin(rx, state.order)
    // Prediction-error filter A(z) = 1 − Σ a_k z^{−k}; envelope gain chosen to match the frame's power.
    const errorFilter = [1, ...coeffs.map((c) => -c)]
    const nfft = 1024
    const spectrum = toFlat(complexAbs(rfft(frame, { n: nfft })))
    const freqs = Array.from({ length: nfft / 2 + 1 }, (_, k) => (k * FS) / nfft)
    const env = responseMagnitude([Math.sqrt(error * N)], errorFilter, nfft / 2 + 1)
    const truth = responseMagnitude([Math.sqrt(error * N)], a, nfft / 2 + 1)
    return {
      freqs,
      spectrum: spectrum.map(db),
      envelope: env.map(db),
      truth: truth.map(db),
      gain: 10 * Math.log10(rx[0] / error),
      stable: reflection.every((k) => Math.abs(k) < 1),
    }
  }, [state.order, state.f0])

  const series = [
    { name: 'frame spectrum |X|', x: r.freqs, y: r.spectrum, muted: true },
    { name: 'true formant envelope', x: r.freqs, y: r.truth, slot: 2, dashed: true },
    { name: `LPC envelope, order ${state.order}`, x: r.freqs, y: r.envelope, slot: 1 },
  ] as const
  const top = Math.max(...r.spectrum)

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [top - 70, top + 10] })
  return (
    <Figure
      title="The LPC spectral envelope"
      state={state}
      caption="A vowel-like frame: a pulse train at f₀ through resonances at 700, 1220 and 2600 Hz, Hamming-windowed. The order-p predictor from the Levinson–Durbin recursion defines an all-pole envelope √E/|A(e^{iω})| that traces the formants and ignores the harmonics. Three resonances need order 6; the prediction gain stops improving there. Very high orders start to fit individual harmonics, especially at high f₀."

      readouts={
        <>
          <Readout label="prediction gain r[0]/E" value={`${formatNumber(r.gain)} dB`} />
          <Readout label="all |kₘ| < 1 (stable)" value={r.stable ? 'yes' : 'no'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}

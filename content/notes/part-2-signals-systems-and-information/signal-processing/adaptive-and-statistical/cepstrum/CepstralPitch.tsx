import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import { complexAbs, realPart, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { lfilter } from 'aifn-compute/signal/filters'
import { getWindow } from 'aifn-compute/signal/windows'

const FS = 16000
const N = 1024
const FORMANTS: [number, number][] = [
  [600, 100],
  [1400, 120],
  [2800, 200],
]

function formantPolynomial(): number[] {
  let a = [1]
  for (const [f, b] of FORMANTS) {
    const r = Math.exp((-Math.PI * b) / FS)
    const th = (2 * Math.PI * f) / FS
    const s = [1, -2 * r * Math.cos(th), r * r]
    const next = new Array(a.length + 2).fill(0)
    a.forEach((ai, i) => s.forEach((sj, j) => (next[i + j] += ai * sj)))
    a = next
  }
  return a
}

/**
 * Real cepstrum c[q] = IDFT(log|X[k]|) of a voiced frame. The source's harmonics give a peak at the pitch period;
 * keeping only low quefrencies (liftering) recovers the smooth vocal-tract envelope.
 */
export function CepstralPitch() {
  const state = useFigureState({
    f0: int(200, { min: 80, max: 400, step: 5, label: 'fundamental f₀ (Hz)' }),
    lifter: int(30, { min: 5, max: 120, step: 1, label: 'lifter cut-off L (samples)' }),
  })

  const r = useMemo(() => {
    const period = FS / state.f0
    const excitation = Array.from({ length: 4000 }, (_, n) =>
      Math.floor(n / period) !== Math.floor((n - 1) / period) ? 1 : 0,
    )
    const voiced = toFlat(lfilter({ b: [1], a: formantPolynomial() }, excitation).y as Tensor).slice(2000, 2000 + N)
    const w = toFlat(getWindow('hann', N, { periodic: true }))
    const X = fft(voiced.map((v, n) => v * w[n]))
    const logMag = toFlat(complexAbs(X)).map((m) => Math.log(m + 1e-9))
    const c = toFlat(realPart(ifft(logMag)))
    // Pitch search over quefrencies 2.5–12.5 ms (80–400 Hz).
    const lo = Math.round(0.0025 * FS)
    const hi = Math.round(0.0125 * FS)
    let peak = lo
    for (let q = lo; q <= hi; q++) if (c[q] > c[peak]) peak = q
    // Lifter: keep quefrencies below L (and their mirror), transform back to a smoothed log spectrum.
    const kept = Array.from(c, (v, q) => (q < state.lifter || q > N - state.lifter ? v : 0))
    const smooth = toFlat(realPart(fft(kept)))
    const half = N / 2 + 1
    const toDb = 20 / Math.LN10
    return {
      freqs: Array.from({ length: half }, (_, k) => (k * FS) / N),
      logDb: logMag.slice(0, half).map((v) => v * toDb),
      smoothDb: Array.from(smooth.slice(0, half), (v) => v * toDb),
      quefrency: Array.from({ length: hi + 20 }, (_, q) => (1000 * q) / FS),
      cep: Array.from(c.slice(0, hi + 20)),
      peak,
    }
  }, [state.f0, state.lifter])

  const spectrum = [
    { name: 'log |X| (dB)', x: r.freqs, y: r.logDb, muted: true },
    { name: `liftered, L = ${state.lifter}`, x: r.freqs, y: r.smoothDb, slot: 1 },
  ] as const
  const cepstrum = [
    { name: 'real cepstrum c[q]', x: r.quefrency.slice(1), y: r.cep.slice(1), slot: 0 },
    { name: 'pitch peak', x: [(1000 * r.peak) / FS], y: [r.cep[r.peak]], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'frequency (Hz)', range: [0, FS / 2] })
  const yAxis = useAxis({ label: 'dB', hold: 'union' })
  const xAxis2 = useAxis({ label: 'quefrency (ms)', hold: 'union' })
  const yAxis2 = useAxis({ label: 'c[q]', hold: 'union' })
  return (
    <Figure
      title="Pitch and envelope from the cepstrum"
      state={state}
      caption="A voiced frame: a pulse train at f₀ through three formant resonances. Left: its log spectrum, harmonics riding on the formant envelope, and the envelope recovered by keeping only quefrencies below L. Right: the real cepstrum. The harmonics' regular spacing f₀ in frequency becomes a peak at quefrency 1/f₀ in the cepstrum; low quefrencies hold the slowly varying envelope. Too large an L lets the harmonic ripple back in."

      readouts={
        <>
          <Readout label="true period" value={`${formatNumber(1000 / state.f0)} ms`} />
          <Readout label="cepstral peak" value={`${formatNumber((1000 * r.peak) / FS)} ms`} />
          <Readout label="estimated f₀" value={`${formatNumber(FS / r.peak)} Hz`} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...spectrum[0]} />
          <Curve {...spectrum[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...cepstrum[0]} />
          <Points {...cepstrum[1]} />
        </Plot>
      </div>
    </Figure>
  )
}

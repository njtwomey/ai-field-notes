import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { complexAbs, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { freqz } from 'aifn-compute/signal/filters'
import { transferFunction } from 'aifn-compute/systems'
import { solve, toeplitz } from '../_shared/stat'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 400
const SHOW = 120
const OMEGA_COUNT = 256

/** Decibels, 20 log₁₀ of a magnitude, floored at −200 dB so zeros stay finite. */
const db = (m: number) => Math.max(-200, 20 * Math.log10(Math.max(m, 1e-300)))

/**
 * FIR Wiener filter for an AR(1) signal in white noise. The taps solve R w = p with R the Toeplitz autocorrelation of
 * the noisy input and p its cross-correlation with the clean signal, both known in closed form for this model.
 */
export function WienerDenoising() {
  const state = useFigureState({
    pole: float(0.9, { min: 0, max: 0.98, step: 0.01, label: 'AR pole a' }),
    snr: int(0, { min: -10, max: 20, step: 1, label: 'SNR (dB)' }),
    taps: int(5, { min: 1, max: 20, step: 1, label: 'filter taps M' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })

  const r = useMemo(() => {
    const a = state.pole
    const varS = 1 / (1 - a * a) // AR(1) driven by unit-variance white noise
    const varV = varS / 10 ** (state.snr / 10)
    // Signal and noise with fixed seeds, so sliders deform the same realisation.
    const g = stream(state.seed)
    const s: number[] = []
    let prev = normal(g) * Math.sqrt(varS)
    for (let n = 0; n < N; n++) {
      prev = a * prev + normal(g)
      s.push(prev)
    }
    const x = s.map((v) => v + Math.sqrt(varV) * normal(g))
    // Wiener–Hopf: r_x[k] = r_s[k] + σ_v² δ[k], p[k] = r_s[k] = σ_s² a^k.
    const M = state.taps
    const rs = Array.from({ length: M }, (_, k) => varS * a ** k)
    const R = toeplitz(rs).map((row, i) => row.map((v, j) => v + (i === j ? varV : 0)))
    const w = solve(R, rs)
    const mse = varS - w.reduce((acc, wk, k) => acc + wk * rs[k], 0)
    const y = x.map((_, n) => w.reduce((acc, wk, k) => acc + (n - k >= 0 ? wk * x[n - k] : 0), 0))
    const empirical = (e: number[]) => e.slice(M).reduce((acc, v) => acc + v * v, 0) / (N - M)
    // Noncausal Wiener gain S_s / (S_s + S_v), and the FIR filter's response.
    const h = freqz(transferFunction(w, [1], { dt: 1 }), { n: OMEGA_COUNT, includeNyquist: true, axis: 'rad/sample' })
    const omega = toFlat(h.f)
    const magnitude = toFlat(complexAbs(h.values) as Tensor)
    const ideal = omega.map((om) => {
      const ss = 1 / (1 + a * a - 2 * a * Math.cos(om))
      return ss / (ss + varV)
    })
    return {
      s,
      x,
      y,
      w,
      mse,
      varS,
      varV,
      empNoisy: empirical(x.map((v, n) => v - s[n])),
      empFiltered: empirical(y.map((v, n) => v - s[n])),
      omega,
      magnitude,
      ideal,
    }
  }, [state.pole, state.snr, state.taps, state.seed])

  const t = Array.from({ length: SHOW }, (_, n) => n + 200)
  const slice = (v: number[]) => t.map((n) => v[n])
  const timeSeries = [
    { name: 'noisy input x[n]', x: t, y: slice(r.x), muted: true },
    { name: 'clean signal s[n]', x: t, y: slice(r.s), slot: 0, dashed: true },
    { name: 'Wiener output y[n]', x: t, y: slice(r.y), slot: 1 },
  ] as const
  const response = [
    { name: 'noncausal Wiener gain', x: r.omega, y: r.ideal.map((v) => db(v)), slot: 2, dashed: true },
    {
      name: `FIR Wiener filter, ${state.taps} taps`,
      x: r.omega,
      y: r.magnitude.map((v) => db(v)),
      slot: 1,
    },
  ] as const

  const xAxis = useAxis({ label: 'n', hold: 'union' })
  const yAxis = useAxis({ label: 'amplitude', hold: 'union' })
  const xAxis2 = useAxis({ label: 'ω (rad/sample)', range: [0, Math.PI] })
  const yAxis2 = useAxis({ label: 'gain (dB)', range: [-40, 5] })
  return (
    <Figure
      title="Wiener filtering an AR(1) signal in white noise"
      state={state}
      caption="The signal is s[n] = a s[n−1] + w[n] with unit-variance white w; the observation adds white noise at the chosen SNR. The FIR Wiener taps solve the Wiener–Hopf equations R w = p using the model's exact correlations. Left: a stretch of the clean, noisy and filtered signals. Right: the filter's magnitude response against the ideal noncausal gain S_s/(S_s + S_v), which passes frequencies where the signal dominates the noise. More taps, a stronger pole or a higher SNR all lower the error."

      readouts={
        <>
          <Readout label="σ_s², σ_v²" value={`${formatNumber(r.varS)}, ${formatNumber(r.varV)}`} />
          <Readout label="theoretical MSE" value={formatNumber(r.mse)} />
          <Readout label="empirical MSE, filtered" value={formatNumber(r.empFiltered)} />
          <Readout label="empirical MSE, unfiltered" value={formatNumber(r.empNoisy)} />
          <Readout label="w₀, w₁, w₂" value={r.w.slice(0, 3).map(formatNumber).join(', ')} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...timeSeries[0]} />
          <Curve {...timeSeries[1]} />
          <Curve {...timeSeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...response[0]} />
          <Curve {...response[1]} />
        </Plot>
      </div>
    </Figure>
  )
}

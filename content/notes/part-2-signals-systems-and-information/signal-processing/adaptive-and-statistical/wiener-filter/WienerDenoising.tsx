import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { db, freqz } from '@/lib/dsp'
import { rng } from '@/lib/math'
import { solve, toeplitz } from '../_shared/stat'

const N = 400
const SHOW = 120
const OMEGA_COUNT = 256

/**
 * FIR Wiener filter for an AR(1) signal in white noise. The taps solve R w = p with R the Toeplitz autocorrelation of
 * the noisy input and p its cross-correlation with the clean signal, both known in closed form for this model.
 */
export function WienerDenoising() {
  const pole = useParam(0.9, { min: 0, max: 0.98, step: 0.01 })
  const snr = useParam(0, { min: -10, max: 20, step: 1 })
  const taps = useParam(5, { min: 1, max: 20, step: 1 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const a = pole.value
    const varS = 1 / (1 - a * a) // AR(1) driven by unit-variance white noise
    const varV = varS / 10 ** (snr.value / 10)
    // Signal and noise with fixed seeds, so sliders deform the same realisation.
    const g = rng(seed.value)
    const s: number[] = []
    let prev = g.normal() * Math.sqrt(varS)
    for (let n = 0; n < N; n++) {
      prev = a * prev + g.normal()
      s.push(prev)
    }
    const x = s.map((v) => v + Math.sqrt(varV) * g.normal())
    // Wiener–Hopf: r_x[k] = r_s[k] + σ_v² δ[k], p[k] = r_s[k] = σ_s² a^k.
    const M = taps.value
    const rs = Array.from({ length: M }, (_, k) => varS * a ** k)
    const R = toeplitz(rs).map((row, i) => row.map((v, j) => v + (i === j ? varV : 0)))
    const w = solve(R, rs)
    const mse = varS - w.reduce((acc, wk, k) => acc + wk * rs[k], 0)
    const y = x.map((_, n) => w.reduce((acc, wk, k) => acc + (n - k >= 0 ? wk * x[n - k] : 0), 0))
    const empirical = (e: number[]) => e.slice(M).reduce((acc, v) => acc + v * v, 0) / (N - M)
    // Noncausal Wiener gain S_s / (S_s + S_v), and the FIR filter's response.
    const { omega, magnitude } = freqz(w, [1], OMEGA_COUNT)
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
  }, [pole.value, snr.value, taps.value, seed.value])

  const t = Array.from({ length: SHOW }, (_, n) => n + 200)
  const slice = (v: number[]) => t.map((n) => v[n])
  const timeSeries: XYSeries[] = [
    { name: 'noisy input x[n]', type: 'line', x: t, y: slice(r.x), muted: true },
    { name: 'clean signal s[n]', type: 'line', x: t, y: slice(r.s), slot: 0, dashed: true },
    { name: 'Wiener output y[n]', type: 'line', x: t, y: slice(r.y), slot: 1 },
  ]
  const response: XYSeries[] = [
    { name: 'noncausal Wiener gain', type: 'line', x: r.omega, y: r.ideal.map((v) => db(v)), slot: 2, dashed: true },
    {
      name: `FIR Wiener filter, ${taps.value} taps`,
      type: 'line',
      x: r.omega,
      y: r.magnitude.map((v) => db(v)),
      slot: 1,
    },
  ]

  return (
    <Interactive
      title="Wiener filtering an AR(1) signal in white noise"
      caption="The signal is s[n] = a s[n−1] + w[n] with unit-variance white w; the observation adds white noise at the chosen SNR. The FIR Wiener taps solve the Wiener–Hopf equations R w = p using the model's exact correlations. Left: a stretch of the clean, noisy and filtered signals. Right: the filter's magnitude response against the ideal noncausal gain S_s/(S_s + S_v), which passes frequencies where the signal dominates the noise. More taps, a stronger pole or a higher SNR all lower the error."
      controls={
        <>
          <ParamSlider label="AR pole a" param={pole} />
          <ParamSlider label="SNR (dB)" param={snr} />
          <ParamSlider label="filter taps M" param={taps} withArrows />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
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
        <XYChart series={timeSeries} xLabel="n" yLabel="amplitude" height={300} />
        <XYChart
          series={response}
          xLabel="ω (rad/sample)"
          yLabel="gain (dB)"
          xRange={[0, Math.PI]}
          yRange={[-40, 5]}
          height={300}
        />
      </div>
    </Interactive>
  )
}

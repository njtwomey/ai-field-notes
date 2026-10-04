import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { halfSpectrum, whiteNoise, windowOf } from '../_shared/spectra'

const N = 8192
const SEG = 128

/**
 * Welch estimates of the cross-spectrum and coherence between white noise x and y = x delayed by d samples plus
 * independent noise. Coherence is 1/(1 + noise variance) at every frequency; the cross-spectrum's phase is −ωd.
 */
export function CoherenceDemo() {
  const state = useFigureState({
    delay: int(5, { min: 0, max: 20, step: 1, label: 'delay d (samples)', format: (v) => String(v) }),
    noise: float(0.5, { min: 0, max: 3, step: 0.1, label: 'noise standard deviation σ' }),
    overlap: float(0.5, {
      min: 0,
      max: 0.75,
      step: 0.25,
      label: 'segment overlap',
      format: (v) => `${Math.round(v * 100)}%`,
    }),
    single: setting(false, 'estimate from a single segment'),
  })

  const r = useMemo(() => {
    const x = whiteNoise(N + 32, 11)
    const e = whiteNoise(N + 32, 12)
    const y = x.map((_, n) => (n >= state.delay ? x[n - state.delay] : 0) + state.noise * e[n])
    const w = windowOf('hann', SEG)
    const half = SEG / 2 + 1
    const sxx = new Array<number>(half).fill(0)
    const syy = new Array<number>(half).fill(0)
    const sxyRe = new Array<number>(half).fill(0)
    const sxyIm = new Array<number>(half).fill(0)
    const step = Math.max(1, Math.round(SEG * (1 - state.overlap)))
    // With "one segment", only the first segment is used: coherence is then identically 1.
    const last = state.single ? 0 : N - SEG
    let count = 0
    for (let s = 32; s <= 32 + last; s += step) {
      const X = halfSpectrum(
        x.slice(s, s + SEG).map((v, i) => v * w[i]),
        SEG,
      )
      const Y = halfSpectrum(
        y.slice(s, s + SEG).map((v, i) => v * w[i]),
        SEG,
      )
      for (let k = 0; k < half; k++) {
        sxx[k] += X.re[k] ** 2 + X.im[k] ** 2
        syy[k] += Y.re[k] ** 2 + Y.im[k] ** 2
        // S_xy = conj(X) · Y, so that y = h ∗ x gives S_xy = H S_xx.
        sxyRe[k] += X.re[k] * Y.re[k] + X.im[k] * Y.im[k]
        sxyIm[k] += X.re[k] * Y.im[k] - X.im[k] * Y.re[k]
      }
      count++
    }
    const omega = Array.from({ length: half }, (_, k) => (2 * k) / SEG)
    const coherence = omega.map((_, k) => (sxyRe[k] ** 2 + sxyIm[k] ** 2) / (sxx[k] * syy[k]))
    const phase = omega.map((_, k) => Math.atan2(sxyIm[k], sxyRe[k]))
    const mean = coherence.slice(1, -1).reduce((s, v) => s + v, 0) / (half - 2)
    return { omega, coherence, phase, mean, count }
  }, [state.delay, state.noise, state.overlap, state.single])

  const theory = 1 / (1 + state.noise ** 2)
  const wrap = (p: number) => Math.atan2(Math.sin(p), Math.cos(p))
  const coh = [
    { name: 'estimated coherence', x: r.omega, y: r.coherence, slot: 0 },
    { name: 'theory 1/(1 + σ²)', x: [0, 1], y: [theory, theory], slot: 1, dashed: true },
  ] as const
  const ph = [
    { name: 'estimated phase of S_xy', x: r.omega, y: r.phase, slot: 0 },
    {
      name: 'theory −ωd (wrapped)',
      x: r.omega,
      y: r.omega.map((w) => wrap(-w * Math.PI * state.delay)),
      slot: 1,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'coherence', range: [0, 1.05] })
  const xAxis2 = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'phase (rad)', range: [-3.3, 3.3] })
  return (
    <Figure
      title="Coherence and cross-phase"
      state={state}
      caption="White noise x and y = x delayed by d samples plus independent noise of standard deviation σ. Welch-averaged cross-spectra give a coherence near 1/(1 + σ²) at every frequency, the fraction of y's power linearly explained by x, and a cross-spectrum phase that falls linearly with slope −d, the delay. Estimate from one segment and the coherence is exactly 1 everywhere, whatever the noise: coherence needs averaging."

      readouts={
        <>
          <Readout label="segments averaged" value={r.count} />
          <Readout label="mean estimated coherence" value={formatNumber(r.mean)} />
          <Readout label="theory" value={formatNumber(theory)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...coh[0]} />
          <Curve {...coh[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={240}>
          <Points {...ph[0]} />
          <Curve {...ph[1]} />
        </Plot>
      </div>
    </Figure>
  )
}

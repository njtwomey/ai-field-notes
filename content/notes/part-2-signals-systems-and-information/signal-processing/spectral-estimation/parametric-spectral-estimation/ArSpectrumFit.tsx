import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { arProcess, arSpectrum, autocovariance, burg, levinsonDurbin, periodogram, powerDb } from '../_shared/spectra'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const OMEGA = toFlat(linspace(0, Math.PI, 513))
// AR(4) with two sharp peaks close together at 0.20π and 0.26π (pole radius 0.97).
const TRUE_A = (() => {
  let p = [1]
  for (const theta of [0.2 * Math.PI, 0.26 * Math.PI]) {
    const q = [1, -2 * 0.97 * Math.cos(theta), 0.97 * 0.97]
    const next = new Array<number>(p.length + 2).fill(0)
    p.forEach((pv, i) => q.forEach((qv, j) => (next[i + j] += pv * qv)))
    p = next
  }
  return p.slice(1).map((v) => -v)
})()
const TRUE_S = arSpectrum(TRUE_A, 1, OMEGA)

/** AR spectra fitted by Yule–Walker or Burg at a chosen order, against the periodogram and the true AR(4) spectrum. */
export function ArSpectrumFit() {
  const state = useFigureState({
    n: int(128, { min: 32, max: 1024, step: 32, label: 'samples N', format: (v) => String(v) }),
    order: int(4, { min: 1, max: 24, step: 1, label: 'model order p', format: (v) => String(v) }),
    method: choice<'burg' | 'yw'>(
      [
        { value: 'burg', label: 'Burg' },
        { value: 'yw', label: 'Yule–Walker' },
      ],
      'burg',
      { label: 'method' },
    ),
    seed: int(5, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const x = arProcess(TRUE_A, state.n, state.seed)
    const acov = autocovariance(x, 24)
    const fit = (p: number) => (state.method === 'burg' ? burg(x, p) : levinsonDurbin(acov, p))
    const chosen = fit(state.order)
    // AIC(p) = N log σ̂²_p + 2p over orders 1..24.
    const aic = Array.from({ length: 24 }, (_, i) => state.n * Math.log(fit(i + 1).sigma2) + 2 * (i + 1))
    const best = aic.indexOf(Math.min(...aic)) + 1
    const raw = periodogram(x, 'hann', 1024)
    return {
      estimate: arSpectrum(chosen.a, chosen.sigma2, OMEGA),
      raw,
      best,
      sigma2: chosen.sigma2,
    }
  }, [state.n, state.order, state.method, state.seed])

  const series = [
    {
      name: 'periodogram (Hann)',
      x: r.raw.omega.map((w) => w / Math.PI),
      y: r.raw.psd.map((v) => powerDb(v)),
      muted: true,
    },
    {
      name: `AR(${state.order}) estimate`,
      x: OMEGA.map((w) => w / Math.PI),
      y: r.estimate.map((v) => powerDb(v)),
      slot: 0,
    },
    {
      name: 'true AR(4) spectrum',
      x: OMEGA.map((w) => w / Math.PI),
      y: TRUE_S.map((v) => powerDb(v)),
      slot: 1,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'power (dB)', range: [-30, 70] })
  return (
    <Figure
      title="Fitting an AR model to estimate a spectrum"
      state={state}
      caption="An AR(4) process with two sharp peaks 0.06π apart. The parametric estimate fits an AR(p) model to N samples and evaluates its spectrum σ̂² / |1 − Σ âₖ e^{−iωk}|². With short records it resolves the peaks that the periodogram (grey) smears together. Too low an order merges the peaks; too high an order adds spurious ones. Burg's method is usually better than Yule–Walker on short records."

      readouts={
        <>
          <Readout label="order chosen by AIC" value={r.best} />
          <Readout label="estimated driving variance σ̂²" value={formatNumber(r.sigma2)} />
          <Readout label="true σ²" value="1" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
      </Plot>
    </Figure>
  )
}

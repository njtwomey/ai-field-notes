import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { arProcess, arSpectrum, autocovariance, burg, levinsonDurbin, periodogram, powerDb } from '../_shared/spectra'

const OMEGA = linspace(0, Math.PI, 513)
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
  const n = useParam(128, { min: 32, max: 1024, step: 32 })
  const order = useParam(4, { min: 1, max: 24, step: 1 })
  const [method, setMethod] = useState<'burg' | 'yw'>('burg')
  const seed = useParam(5, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const x = arProcess(TRUE_A, n.value, seed.value)
    const acov = autocovariance(x, 24)
    const fit = (p: number) => (method === 'burg' ? burg(x, p) : levinsonDurbin(acov, p))
    const chosen = fit(order.value)
    // AIC(p) = N log σ̂²_p + 2p over orders 1..24.
    const aic = Array.from({ length: 24 }, (_, i) => n.value * Math.log(fit(i + 1).sigma2) + 2 * (i + 1))
    const best = aic.indexOf(Math.min(...aic)) + 1
    const raw = periodogram(x, 'hann', 1024)
    return {
      estimate: arSpectrum(chosen.a, chosen.sigma2, OMEGA),
      raw,
      best,
      sigma2: chosen.sigma2,
    }
  }, [n.value, order.value, method, seed.value])

  const series: XYSeries[] = [
    {
      name: 'periodogram (Hann)',
      type: 'line',
      x: r.raw.omega.map((w) => w / Math.PI),
      y: r.raw.psd.map((v) => powerDb(v)),
      muted: true,
    },
    {
      name: `AR(${order.value}) estimate`,
      type: 'line',
      x: OMEGA.map((w) => w / Math.PI),
      y: r.estimate.map((v) => powerDb(v)),
      slot: 0,
    },
    {
      name: 'true AR(4) spectrum',
      type: 'line',
      x: OMEGA.map((w) => w / Math.PI),
      y: TRUE_S.map((v) => powerDb(v)),
      slot: 1,
      dashed: true,
    },
  ]

  return (
    <Interactive
      title="Fitting an AR model to estimate a spectrum"
      caption="An AR(4) process with two sharp peaks 0.06π apart. The parametric estimate fits an AR(p) model to N samples and evaluates its spectrum σ̂² / |1 − Σ âₖ e^{−iωk}|². With short records it resolves the peaks that the periodogram (grey) smears together. Too low an order merges the peaks; too high an order adds spurious ones. Burg's method is usually better than Yule–Walker on short records."
      controls={
        <>
          <ParamSlider label="samples N" param={n} format={(v) => String(v)} withArrows />
          <ParamSlider label="model order p" param={order} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="method"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'burg', label: 'Burg' },
              { value: 'yw', label: 'Yule–Walker' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="order chosen by AIC" value={r.best} />
          <Readout label="estimated driving variance σ̂²" value={formatNumber(r.sigma2)} />
          <Readout label="true σ²" value="1" />
        </>
      }
    >
      <XYChart series={series} xLabel="ω / π" yLabel="power (dB)" xRange={[0, 1]} yRange={[-30, 70]} height={320} />
    </Interactive>
  )
}

import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

/** Data N(0, 1) against a Gaussian generator N(m, s²); D* = p_data / (p_data + p_g) and JS by numerical integration. */
const X = linspace(-6, 6, 481)
const DX = X[1] - X[0]
const normal = (x: number, m: number, s: number) =>
  Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI))
const P_DATA = X.map((x) => normal(x, 0, 1))

const xlogx = (p: number, q: number) => (p > 1e-300 ? p * Math.log(p / q) : 0)

export function OptimalDiscriminator() {
  const m = useParam(2, { min: -3, max: 3, step: 0.05 })
  const s = useParam(0.6, { min: 0.4, max: 2, step: 0.05 })

  const view = useMemo(() => {
    const pg = X.map((x) => normal(x, m.value, s.value))
    const d = X.map((_, i) => {
      const tot = P_DATA[i] + pg[i]
      return tot > 1e-300 ? P_DATA[i] / tot : 0.5
    })
    let js = 0
    for (let i = 0; i < X.length; i++) {
      const mix = 0.5 * (P_DATA[i] + pg[i])
      js += 0.5 * (xlogx(P_DATA[i], mix) + xlogx(pg[i], mix)) * DX
    }
    const series: XYSeries[] = [
      { name: 'p_data', type: 'line', x: X, y: P_DATA, slot: 0, area: true },
      { name: 'p_g (generator)', type: 'line', x: X, y: pg, slot: 1, area: true },
      { name: 'optimal discriminator D*(x)', type: 'line', x: X, y: d, emphasis: true },
    ]
    return { series, js }
  }, [m.value, s.value])

  return (
    <Interactive
      title="The optimal discriminator and the Jensen–Shannon divergence"
      caption="Data are N(0, 1) and the generator outputs N(m, s²). For a fixed generator the best discriminator is D*(x) = p_data(x) / (p_data(x) + p_g(x)). It is 1/2 where the densities are equal and saturates at 0 or 1 where only one has mass. The GAN value at D* is 2 JS − log 4, which reaches its minimum −log 4 ≈ −1.386 only when p_g = p_data."
      controls={
        <>
          <ParamSlider label="generator mean m" param={m} />
          <ParamSlider label="generator std s" param={s} />
        </>
      }
      readout={
        <>
          <Readout label="JS(p_data, p_g) (nats)" value={formatNumber(view.js)} />
          <Readout label="V(D*, G) = 2 JS − log 4" value={formatNumber(2 * view.js - Math.log(4))} />
        </>
      }
    >
      <XYChart series={view.series} xLabel="x" yLabel="density / probability" xRange={[-6, 6]} yRange={[0, 1.05]} />
    </Interactive>
  )
}

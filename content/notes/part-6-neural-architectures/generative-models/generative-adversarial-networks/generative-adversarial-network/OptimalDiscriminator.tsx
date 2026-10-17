import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

/** Data N(0, 1) against a Gaussian generator N(m, s²); D* = p_data / (p_data + p_g) and JS by numerical integration. */
const X = toFlat(linspace(-6, 6, 481))
const DX = X[1] - X[0]
const normal = (x: number, m: number, s: number) =>
  Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI))
const P_DATA = X.map((x) => normal(x, 0, 1))

const xlogx = (p: number, q: number) => (p > 1e-300 ? p * Math.log(p / q) : 0)

export function OptimalDiscriminator() {
  const state = useFigureState({
    m: float(2, { min: -3, max: 3, step: 0.05, label: 'generator mean m' }),
    s: float(0.6, { min: 0.4, max: 2, step: 0.05, label: 'generator std s' }),
  })

  const view = useMemo(() => {
    const pg = X.map((x) => normal(x, state.m, state.s))
    const d = X.map((_, i) => {
      const tot = P_DATA[i] + pg[i]
      return tot > 1e-300 ? P_DATA[i] / tot : 0.5
    })
    let js = 0
    for (let i = 0; i < X.length; i++) {
      const mix = 0.5 * (P_DATA[i] + pg[i])
      js += 0.5 * (xlogx(P_DATA[i], mix) + xlogx(pg[i], mix)) * DX
    }
    const series = [
      { name: 'p_data', x: X, y: P_DATA, slot: 0 },
      { name: 'p_g (generator)', x: X, y: pg, slot: 1 },
      { name: 'optimal discriminator D*(x)', x: X, y: d, emphasis: true },
    ] as const
    return { series, js }
  }, [state.m, state.s])

  const xAxis = useAxis({ label: 'x', range: [-6, 6] })
  const yAxis = useAxis({ label: 'density / probability', range: [0, 1.05] })
  return (
    <Figure
      title="The optimal discriminator and the Jensen–Shannon divergence"
      state={state}
      caption="Data are N(0, 1) and the generator outputs N(m, s²). For a fixed generator the best discriminator is D*(x) = p_data(x) / (p_data(x) + p_g(x)). It is 1/2 where the densities are equal and saturates at 0 or 1 where only one has mass. The GAN value at D* is 2 JS − log 4, which reaches its minimum −log 4 ≈ −1.386 only when p_g = p_data."

      readouts={
        <>
          <Readout label="JS(p_data, p_g) (nats)" value={formatNumber(view.js)} />
          <Readout label="V(D*, G) = 2 JS − log 4" value={formatNumber(2 * view.js - Math.log(4))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Area {...view.series[0]} />
        <Area {...view.series[1]} />
        <Curve {...view.series[2]} />
      </Plot>
    </Figure>
  )
}

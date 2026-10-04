import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * A one-dimensional linear-Gaussian VAE with everything in closed form: prior p(z) = N(0, 1), decoder
 * p(x | z) = N(W z, S²), and for each x the encoder that maximises E_q[log p(x | z)] − β KL(q ‖ p).
 * That maximiser is q(z | x) ∝ p(z) p(x | z)^{1/β}, a Gaussian with precision 1 + W²/(β S²).
 */
const W = 2
const S = 0.5
const XS = [-3, -1.5, 0, 1.5, 3]
const Z = toFlat(linspace(-3.5, 3.5, 281))
const LOG_2PI = Math.log(2 * Math.PI)

const gauss = (z: number, m: number, v: number) => Math.exp(-((z - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)

function encoder(x: number, beta: number) {
  const precision = 1 + (W * W) / (beta * S * S)
  const v = 1 / precision
  const m = ((W * x) / (beta * S * S)) * v
  const rate = 0.5 * (m * m + v - 1 - Math.log(v))
  const distortion = ((x - W * m) ** 2 + W * W * v) / (2 * S * S) + 0.5 * (LOG_2PI + Math.log(S * S))
  return { m, v, rate, distortion }
}

export function LatentPosteriors({ initialLogBeta = 0 }: { initialLogBeta?: number }) {
  const state = useFigureState({
    logBeta: float(initialLogBeta, {
      min: -1,
      max: 1.5,
      step: 0.05,
      label: 'KL weight β',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
  })
  const beta = 10 ** state.logBeta

  const view = useMemo(() => {
    const encs = XS.map((x) => encoder(x, beta))
    const series: SeriesSpec[] = [
      { name: 'prior p(z)', type: 'line', x: Z, y: Z.map((z) => gauss(z, 0, 1)), emphasis: true, dashed: true },
      ...encs.map((e, i): SeriesSpec => ({
        name: `q(z | x = ${XS[i]})`,
        type: 'line',
        x: Z,
        y: Z.map((z) => gauss(z, e.m, e.v)),
        slot: i,
      })),
    ]
    const mean = (f: (e: (typeof encs)[number]) => number) => encs.reduce((s, e) => s + f(e), 0) / encs.length
    const rate = mean((e) => e.rate)
    const distortion = mean((e) => e.distortion)
    const evidence = XS.reduce((s, x) => s - 0.5 * (LOG_2PI + Math.log(W * W + S * S) + (x * x) / (W * W + S * S)), 0)
    return { series, rate, distortion, elbo: -distortion - rate, logEvidence: evidence / XS.length }
  }, [beta])

  const xAxis = useAxis({ label: 'latent z', range: [-3.5, 3.5] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Encoder distributions and the rate–distortion trade-off"
      state={state}
      caption="A one-dimensional VAE with prior N(0, 1) and a fixed linear decoder p(x | z) = N(2z, 0.5²). Each coloured curve is the optimal encoder q(z | x) for one data point when the KL term has weight β. At β = 1 each q is the exact posterior and the ELBO equals log p(x). Small β gives narrow, well-separated codes (high rate, low distortion); large β pulls every q towards the prior."

      readouts={
        <>
          <Readout label="rate: mean KL(q ‖ p) (nats)" value={formatNumber(view.rate)} />
          <Readout label="distortion: mean −E_q log p(x | z)" value={formatNumber(view.distortion)} />
          <Readout label="mean ELBO" value={formatNumber(view.elbo)} />
          <Readout label="mean log p(x)" value={formatNumber(view.logEvidence)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(view.series)}
      </Plot>
    </Figure>
  )
}

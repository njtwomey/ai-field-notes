import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { basis, fitRvm } from './rvm'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const GRID = toFlat(linspace(-15, 15, 241))
const X_RANGE: [number, number] = [-15, 15]
const Y_RANGE: [number | undefined, number | undefined] = [-0.6, 1.4]
const sinc = (x: number) => (x === 0 ? 1 : Math.sin(x) / x)

/** RVM regression of noisy sinc data: the fit uses only the few training inputs whose bumps survive. */
export function RvmFit() {
  const state = useFigureState({
    n: int(50, { min: 10, max: 100, step: 5, label: 'training points N', format: (v) => String(v) }),
    noise: float(0.1, { min: 0.02, max: 0.4, step: 0.01, label: 'noise sd' }),
    width: float(2, { min: 0.5, max: 4, step: 0.1, label: 'bump width' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'data seed', format: (v) => String(v) }),
  })

  const data = useMemo(() => {
    const g = stream(state.seed)
    const x = Array.from({ length: state.n }, () => -10 + 20 * uniform(g)).sort((a, b) => a - b)
    return { x, y: x.map((v) => sinc(v) + state.noise * normal(g)) }
  }, [state.n, state.noise, state.seed])

  const r = useMemo(() => {
    const fit = fitRvm(data.x, data.y, state.width)
    const phiAt = (x: number) => fit.active.map((j) => basis(j, x, data.x, state.width))
    const mean: number[] = []
    const sd: number[] = []
    for (const x of GRID) {
      const f = phiAt(x)
      mean.push(f.reduce((s, v, a) => s + v * fit.mean[a], 0))
      const sf = fit.covariance.map((row) => row.reduce((s, v, b) => s + v * f[b], 0))
      sd.push(Math.sqrt(1 / fit.beta + f.reduce((s, v, a) => s + v * sf[a], 0)))
    }
    const vectors = fit.active.filter((j) => j > 0).map((j) => j - 1)
    return { fit, mean, sd, vectors }
  }, [data, state.width])

  const series = [
    { name: 'sin x / x', x: GRID, y: GRID.map(sinc), muted: true, dashed: true },
    {
      name: 'predictive + 2 sd',
      x: GRID,
      y: r.mean.map((m, i) => m + 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'predictive − 2 sd',
      x: GRID,
      y: r.mean.map((m, i) => m - 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'predictive mean', x: GRID, y: r.mean, slot: 0 },
    { name: 'data', x: data.x, y: data.y, slot: 1 },
    {
      name: 'relevance vectors',
      x: r.vectors.map((i) => data.x[i]),
      y: r.vectors.map((i) => data.y[i]),
      emphasis: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'y', range: Y_RANGE })
  return (
    <Figure
      title="A sparse kernel regression"
      state={state}
      caption="Noisy samples of sin x / x fitted by a relevance vector machine with one Gaussian bump of the chosen width per training input, plus a bias. Every weight has its own prior precision, re-estimated from the marginal likelihood until most precisions diverge and their bumps are pruned. The diamonds are the training inputs whose bumps remain, the relevance vectors. They are a small fraction of the data. The noise level is estimated too. Beyond the data, at both ends, the band shrinks to the noise level: the model has no basis functions there, so its only uncertainty is the noise."

      readouts={
        <>
          <Readout label="relevance vectors" value={`${r.vectors.length} of ${data.x.length}`} />
          <Readout label="bias kept" value={r.fit.active.includes(0) ? 'yes' : 'no'} />
          <Readout label="estimated noise sd 1/√β" value={formatNumber(1 / Math.sqrt(r.fit.beta))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Points {...series[4]} />
        <Points {...series[5]} />
      </Plot>
    </Figure>
  )
}

import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { basis, fitRvm } from './rvm'

const GRID = linspace(-15, 15, 241)
const X_RANGE: [number, number] = [-15, 15]
const Y_RANGE: [number | undefined, number | undefined] = [-0.6, 1.4]
const sinc = (x: number) => (x === 0 ? 1 : Math.sin(x) / x)

/** RVM regression of noisy sinc data: the fit uses only the few training inputs whose bumps survive. */
export function RvmFit() {
  const n = useParam(50, { min: 10, max: 100, step: 5 })
  const noise = useParam(0.1, { min: 0.02, max: 0.4, step: 0.01 })
  const width = useParam(2, { min: 0.5, max: 4, step: 0.1 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const data = useMemo(() => {
    const g = rng(seed.value)
    const x = Array.from({ length: n.value }, () => -10 + 20 * g.uniform()).sort((a, b) => a - b)
    return { x, y: x.map((v) => sinc(v) + noise.value * g.normal()) }
  }, [n.value, noise.value, seed.value])

  const r = useMemo(() => {
    const fit = fitRvm(data.x, data.y, width.value)
    const phiAt = (x: number) => fit.active.map((j) => basis(j, x, data.x, width.value))
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
  }, [data, width.value])

  const series: XYSeries[] = [
    { name: 'sin x / x', type: 'line', x: GRID, y: GRID.map(sinc), muted: true, dashed: true },
    {
      name: 'predictive + 2 sd',
      type: 'line',
      x: GRID,
      y: r.mean.map((m, i) => m + 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'predictive − 2 sd',
      type: 'line',
      x: GRID,
      y: r.mean.map((m, i) => m - 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'predictive mean', type: 'line', x: GRID, y: r.mean, slot: 0 },
    { name: 'data', type: 'scatter', x: data.x, y: data.y, slot: 1 },
    {
      name: 'relevance vectors',
      type: 'scatter',
      x: r.vectors.map((i) => data.x[i]),
      y: r.vectors.map((i) => data.y[i]),
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="A sparse kernel regression"
      caption="Noisy samples of sin x / x fitted by a relevance vector machine with one Gaussian bump of the chosen width per training input, plus a bias. Every weight has its own prior precision, re-estimated from the marginal likelihood until most precisions diverge and their bumps are pruned. The diamonds are the training inputs whose bumps remain, the relevance vectors. They are a small fraction of the data. The noise level is estimated too. Beyond the data, at both ends, the band shrinks to the noise level: the model has no basis functions there, so its only uncertainty is the noise."
      controls={
        <>
          <ParamSlider label="training points N" param={n} format={(v) => String(v)} />
          <ParamSlider label="noise sd" param={noise} />
          <ParamSlider label="bump width" param={width} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="relevance vectors" value={`${r.vectors.length} of ${data.x.length}`} />
          <Readout label="bias kept" value={r.fit.active.includes(0) ? 'yes' : 'no'} />
          <Readout label="estimated noise sd 1/√β" value={formatNumber(1 / Math.sqrt(r.fit.beta))} />
        </>
      }
    >
      <XYChart series={series} xLabel="x" yLabel="y" xRange={X_RANGE} yRange={Y_RANGE} height={380} />
    </Interactive>
  )
}

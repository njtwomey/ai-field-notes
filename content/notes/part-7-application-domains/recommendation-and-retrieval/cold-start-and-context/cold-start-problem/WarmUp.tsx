import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

const N_MAX = 60
const NS = Array.from({ length: N_MAX + 1 }, (_, n) => n)
const D = 16 // latent dimension
const TAU2 = 1 // prior variance of an item's latent coordinates (per dimension, scaled)

/**
 * Expected squared error of a new rating prediction for an item after n observed ratings, in a linear-Gaussian model:
 * r = w·v + ε with E[ww^T] = I/d, ε ~ N(0, σ²). A prior N(m, s²I) on v and n ratings from random users give posterior
 * variance 1/(1/s² + n/(dσ²)) per dimension, which is also the expected squared error of w·v̂ for a fresh user.
 */
const posterior = (prior: number, n: number, sigma2: number) => 1 / (1 / prior + n / (D * sigma2))

/** ID-only, content-only and hybrid item embeddings as ratings accumulate for a new item. */
export function WarmUp() {
  const state = useFigureState({
    quality: float(0.6, { min: 0, max: 0.95, step: 0.05, label: 'variance explained by item features' }),
    noise: float(0.5, { min: 0.1, max: 1.5, step: 0.05, label: 'rating noise σ' }),
    count: int(5, { min: 0, max: N_MAX, step: 1, label: 'ratings observed for the item', format: (v) => String(v) }),
  })
  const s2 = (1 - state.quality) * TAU2
  const sigma2 = state.noise ** 2

  const series = useMemo(() => {
    const id = NS.map((n) => Math.sqrt(posterior(TAU2, n, sigma2)))
    const content = NS.map(() => Math.sqrt(s2))
    const hybrid = NS.map((n) => Math.sqrt(posterior(s2, n, sigma2)))
    return [
      { name: 'ID embedding only (collaborative)', x: NS, y: id, slot: 0 },
      { name: 'content features only', x: NS, y: content, slot: 1 },
      { name: 'hybrid: content prior + ID updates', x: NS, y: hybrid, slot: 2 },
    ] as const
  }, [s2, sigma2])

  const n = state.count
  const xAxis = useAxis({ label: 'ratings observed for the new item', range: [0, N_MAX] })
  const yAxis = useAxis({ label: 'expected prediction error (RMS)', range: [0, 1.05] })
  return (
    <Figure
      title="Warming up a new item"
      state={state}
      caption="An idealised linear-Gaussian model of a new item's latent vector. A collaborative ID embedding starts at the prior (it knows nothing) and improves as ratings arrive. A pure content embedding is available at once but never improves, because it does not use the item's ratings; its error is the part of the item's taste profile the features cannot explain. A hybrid starts from the content estimate and refines it with the ratings, so it is best at every count. Raise the feature quality to shrink the content error; step the number of ratings with the arrows."

      readouts={
        <>
          <Readout label="ID-only error" value={formatNumber(series[0].y[n])} />
          <Readout label="content-only error" value={formatNumber(series[1].y[n])} />
          <Readout label="hybrid error" value={formatNumber(series[2].y[n])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle kind="x" at={n} label="n" onDrag={(x) => state.set('count', x)} />
      </Plot>
    </Figure>
  )
}

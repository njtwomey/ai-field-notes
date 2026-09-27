import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

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
  const quality = useParam(0.6, { min: 0, max: 0.95, step: 0.05 })
  const noise = useParam(0.5, { min: 0.1, max: 1.5, step: 0.05 })
  const count = useParam(5, { min: 0, max: N_MAX, step: 1 })
  const s2 = (1 - quality.value) * TAU2
  const sigma2 = noise.value ** 2

  const series = useMemo((): XYSeries[] => {
    const id = NS.map((n) => Math.sqrt(posterior(TAU2, n, sigma2)))
    const content = NS.map(() => Math.sqrt(s2))
    const hybrid = NS.map((n) => Math.sqrt(posterior(s2, n, sigma2)))
    return [
      { name: 'ID embedding only (collaborative)', type: 'line', x: NS, y: id, slot: 0 },
      { name: 'content features only', type: 'line', x: NS, y: content, slot: 1 },
      { name: 'hybrid: content prior + ID updates', type: 'line', x: NS, y: hybrid, slot: 2 },
    ]
  }, [s2, sigma2])

  const n = count.value
  return (
    <Interactive
      title="Warming up a new item"
      caption="An idealised linear-Gaussian model of a new item's latent vector. A collaborative ID embedding starts at the prior (it knows nothing) and improves as ratings arrive. A pure content embedding is available at once but never improves, because it does not use the item's ratings; its error is the part of the item's taste profile the features cannot explain. A hybrid starts from the content estimate and refines it with the ratings, so it is best at every count. Raise the feature quality to shrink the content error; step the number of ratings with the arrows."
      controls={
        <>
          <ParamSlider label="variance explained by item features" param={quality} />
          <ParamSlider label="rating noise σ" param={noise} />
          <ParamSlider label="ratings observed for the item" param={count} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="ID-only error" value={formatNumber(series[0].y[n])} />
          <Readout label="content-only error" value={formatNumber(series[1].y[n])} />
          <Readout label="hybrid error" value={formatNumber(series[2].y[n])} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="ratings observed for the new item"
        yLabel="expected prediction error (RMS)"
        xRange={[0, N_MAX]}
        yRange={[0, 1.05]}
        handles={[{ kind: 'x', at: n, label: 'n', onDrag: (x) => count.set(x) }]}
      />
    </Interactive>
  )
}

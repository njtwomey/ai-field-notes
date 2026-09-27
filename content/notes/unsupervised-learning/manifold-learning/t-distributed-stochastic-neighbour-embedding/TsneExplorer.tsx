import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber } from '@/components/viz'
import { covariance, eigSymmetric } from '../../_shared/linalg'
import { CLUSTER_NAMES, clusters } from './data'
import { affinities, squaredDistances, tsne } from '../../_shared/tsne'

const ITERATIONS = 500
const EVERY = 10
const PERPLEXITIES = [
  { value: '5', label: '5' },
  { value: '15', label: '15' },
  { value: '30', label: '30' },
  { value: '50', label: '50' },
] as const

type Perplexity = (typeof PERPLEXITIES)[number]['value']

export function TsneExplorer() {
  const data = useMemo(() => {
    const { x, labels } = clusters()
    const { vectors } = eigSymmetric(covariance(x))
    const mean = x[0].map((_, m) => x.reduce((s, r) => s + r[m], 0) / x.length)
    const pca = x.map((r) => [0, 1].map((k) => vectors[k].reduce((s, v, m) => s + v * (r[m] - mean[m]), 0)))
    return { x, labels, d2: squaredDistances(x), pca }
  }, [])
  const [perplexity, setPerplexity] = useState<Perplexity>('30')
  const [seed, setSeed] = useState(1)
  const [iteration, setIteration] = useState(ITERATIONS)
  const p = useMemo(() => affinities(data.d2, Number(perplexity)), [data, perplexity])
  const run = useMemo(() => tsne(p, seed, ITERATIONS, EVERY), [p, seed])
  const y = run.snapshots[iteration / EVERY]

  return (
    <Interactive
      title="t-SNE against PCA on four clusters in ten dimensions"
      caption="A and B are tight clusters 3 apart; C is four times as wide; D is tight and 12 away from the rest. PCA (right) keeps these proportions. t-SNE (left) separates every cluster but compresses C's extra width and shrinks the relative gap to D, and the picture changes with perplexity. Scrub the iteration slider to watch early exaggeration (the first 100 iterations) pull each cluster together before the layout spreads out."
      controls={
        <>
          <ParamChoice label="perplexity" value={perplexity} onChange={setPerplexity} options={PERPLEXITIES} />
          <ParamSlider label="random seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
          <ParamSlider
            label="iteration"
            value={iteration}
            onChange={setIteration}
            min={0}
            max={ITERATIONS}
            step={EVERY}
            debounceMs={0}
            withArrows
          />
        </>
      }
      readout={<Readout label="KL(P ‖ Q) at this iteration" value={formatNumber(run.kl[iteration / EVERY])} />}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={340}
          xLabel="t-SNE 1"
          yLabel="t-SNE 2"
          series={[
            {
              name: 't-SNE',
              type: 'scatter',
              x: y.map((v) => v[0]),
              y: y.map((v) => v[1]),
              group: data.labels,
              groupNames: CLUSTER_NAMES,
            },
          ]}
        />
        <XYChart
          height={340}
          xLabel="PC 1"
          yLabel="PC 2"
          series={[
            {
              name: 'PCA',
              type: 'scatter',
              x: data.pca.map((v) => v[0]),
              y: data.pca.map((v) => v[1]),
              group: data.labels,
              groupNames: CLUSTER_NAMES,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}

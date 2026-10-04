import { useMemo } from 'react'
import { choice, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
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
  const state = useFigureState({
    perplexity: choice<Perplexity>(PERPLEXITIES, '30', { label: 'perplexity' }),
    seed: int(1, { min: 1, max: 10, step: 1, label: 'random seed' }),
    iteration: float(ITERATIONS, { min: 0, max: ITERATIONS, step: EVERY, label: 'iteration' }),
  })
  const p = useMemo(() => affinities(data.d2, Number(state.perplexity)), [data, state.perplexity])
  const run = useMemo(() => tsne(p, state.seed, ITERATIONS, EVERY), [p, state.seed])
  const y = run.snapshots[state.iteration / EVERY]

  const xAxis = useAxis({ label: 't-SNE 1', hold: 'union' })
  const yAxis = useAxis({ label: 't-SNE 2', hold: 'union' })
  const xAxis2 = useAxis({ label: 'PC 1', hold: 'union' })
  const yAxis2 = useAxis({ label: 'PC 2', hold: 'union' })
  return (
    <Figure
      title="t-SNE against PCA on four clusters in ten dimensions"
      state={state}
      caption="A and B are tight clusters 3 apart; C is four times as wide; D is tight and 12 away from the rest. PCA (right) keeps these proportions. t-SNE (left) separates every cluster but compresses C's extra width and shrinks the relative gap to D, and the picture changes with perplexity. Scrub the iteration slider to watch early exaggeration (the first 100 iterations) pull each cluster together before the layout spreads out."

      readouts={<Readout label="KL(P ‖ Q) at this iteration" value={formatNumber(run.kl[state.iteration / EVERY])} />}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Points
            name="t-SNE"
            x={y.map((v) => v[0])}
            y={y.map((v) => v[1])}
            group={data.labels}
            groupNames={CLUSTER_NAMES}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Points
            name="PCA"
            x={data.pca.map((v) => v[0])}
            y={data.pca.map((v) => v[1])}
            group={data.labels}
            groupNames={CLUSTER_NAMES}
          />
        </Plot>
      </div>
    </Figure>
  )
}

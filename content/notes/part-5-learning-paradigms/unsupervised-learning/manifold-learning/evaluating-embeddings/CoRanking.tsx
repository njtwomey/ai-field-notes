import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  coranking,
  distances,
  isomap,
  laplacianEigenmaps,
  manifold,
  pca,
  rankQuality,
  ranks,
  tsneEmbedding,
  umap,
  type DatasetId,
  type Rows,
} from '../../_shared/manifold'

const N = 240
const BIN = 10
const K_MAX = 80

const DATASETS = [
  { value: 'swiss-roll', label: 'Swiss roll' },
  { value: 'blobs', label: 'blobs' },
] as const

const METHODS = [
  { value: 'pca', label: 'PCA' },
  { value: 'isomap', label: 'Isomap' },
  { value: 'le', label: 'Laplacian eigenmaps' },
  { value: 'tsne', label: 't-SNE' },
  { value: 'umap', label: 'UMAP' },
] as const
type Method = (typeof METHODS)[number]['value']

function embed(method: Method, x: Rows, d: number[][]): Rows {
  switch (method) {
    case 'pca':
      return pca(x)
    case 'isomap':
      return isomap(d, 6).y
    case 'le':
      return laplacianEigenmaps(d, 6)
    case 'tsne':
      return tsneEmbedding(x, 30)
    case 'umap':
      return umap(d, 10)
  }
}

export function CoRanking() {
  const state = useFigureState({
    dataset: choice<DatasetId>(DATASETS, 'swiss-roll', { label: 'data' }),
    method: choice<Method>(METHODS, 'pca', { label: 'method' }),
    k: int(10, { min: 1, max: K_MAX, step: 1, label: 'K (neighbourhood size)' }),
  })
  const data = useMemo(() => {
    const m = manifold(state.dataset, N)
    const d = distances(m.x)
    return { x: m.x, d, r: ranks(d) }
  }, [state.dataset])
  const result = useMemo(() => {
    const y = embed(state.method, data.x, data.d)
    const q = coranking(data.r, ranks(distances(y)))
    const bins = Math.ceil((N - 1) / BIN)
    const z = Array.from({ length: bins }, (_, a) =>
      Array.from({ length: bins }, (_, b) => {
        let s = 0
        for (let i = a * BIN; i < Math.min((a + 1) * BIN, N - 1); i++)
          for (let j = b * BIN; j < Math.min((b + 1) * BIN, N - 1); j++) s += q[i][j]
        return Math.log10(1 + s)
      }),
    )
    const ks = Array.from({ length: K_MAX }, (_, i) => i + 1)
    const curves = ks.map((kk) => rankQuality(q, kk))
    const centres = Array.from({ length: bins }, (_, b) => b * BIN + BIN / 2)
    return { q, z, ks, curves, centres }
  }, [data, state.method])
  const now = rankQuality(result.q, state.k)

  const xAxis = useAxis({ label: 'rank in embedding' })
  const yAxis = useAxis({ label: 'rank in data' })
  const xAxis2 = useAxis({ label: 'K', hold: 'union' })
  const yAxis2 = useAxis({ label: 'quality', range: [0, 1] })
  return (
    <Figure
      title="Co-ranking matrix and rank-based quality"
      state={state}
      caption="Left: the co-ranking matrix of 240 points, binned in blocks of 10 ranks and coloured by log₁₀(1 + count). Row: rank of a neighbour in the data; column: its rank in the embedding. A perfect embedding puts every pair on the diagonal. Mass below the diagonal in the left columns is intruders (close in the embedding, far in the data); mass to the right in the top rows is extrusions. Right: trustworthiness, continuity and the fraction of K nearest neighbours kept, against K. Drag the vertical line to change K. On the Swiss roll, Isomap, t-SNE and UMAP beat PCA for small K and lose to it for large K: a large neighbourhood in the data includes points on the next layer of the roll, which a correct unrolling moves far away. On the blobs, t-SNE and UMAP keep small neighbourhoods best."

      readouts={
        <>
          <Readout label="trustworthiness T(K)" value={formatNumber(now.trust)} />
          <Readout label="continuity C(K)" value={formatNumber(now.cont)} />
          <Readout label="neighbours kept Q_NX(K)" value={formatNumber(now.qnx)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster
            x={result.centres}
            y={result.centres}
            z={result.z}
            scale={'sequential'}
            valueLabel={'log₁₀(1 + pairs)'}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve name="trustworthiness" x={result.ks} y={result.curves.map((c) => c.trust)} slot={0} />
          <Curve name="continuity" x={result.ks} y={result.curves.map((c) => c.cont)} slot={1} />
          <Curve name="Q_NX (neighbours kept)" x={result.ks} y={result.curves.map((c) => c.qnx)} slot={2} />
          <Handle {...state.handle('k', { label: 'K' })} />
        </Plot>
      </div>
    </Figure>
  )
}

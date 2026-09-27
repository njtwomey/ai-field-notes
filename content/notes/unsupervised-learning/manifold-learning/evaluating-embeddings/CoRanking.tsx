import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
} from '@/components/viz'
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
  const [dataset, setDataset] = useState<DatasetId>('swiss-roll')
  const [method, setMethod] = useState<Method>('pca')
  const k = useParam(10, { min: 1, max: K_MAX, step: 1 })
  const data = useMemo(() => {
    const m = manifold(dataset, N)
    const d = distances(m.x)
    return { x: m.x, d, r: ranks(d) }
  }, [dataset])
  const result = useMemo(() => {
    const y = embed(method, data.x, data.d)
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
  }, [data, method])
  const now = rankQuality(result.q, k.value)

  return (
    <Interactive
      title="Co-ranking matrix and rank-based quality"
      caption="Left: the co-ranking matrix of 240 points, binned in blocks of 10 ranks and coloured by log₁₀(1 + count). Row: rank of a neighbour in the data; column: its rank in the embedding. A perfect embedding puts every pair on the diagonal. Mass below the diagonal in the left columns is intruders (close in the embedding, far in the data); mass to the right in the top rows is extrusions. Right: trustworthiness, continuity and the fraction of K nearest neighbours kept, against K. Drag the vertical line to change K. On the Swiss roll, Isomap, t-SNE and UMAP beat PCA for small K and lose to it for large K: a large neighbourhood in the data includes points on the next layer of the roll, which a correct unrolling moves far away. On the blobs, t-SNE and UMAP keep small neighbourhoods best."
      controls={
        <>
          <ParamChoice label="data" value={dataset} onChange={setDataset} options={DATASETS} />
          <ParamChoice label="method" value={method} onChange={setMethod} options={METHODS} />
          <ParamSlider label="K (neighbourhood size)" param={k} />
        </>
      }
      readout={
        <>
          <Readout label="trustworthiness T(K)" value={formatNumber(now.trust)} />
          <Readout label="continuity C(K)" value={formatNumber(now.cont)} />
          <Readout label="neighbours kept Q_NX(K)" value={formatNumber(now.qnx)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={result.centres}
          y={result.centres}
          z={result.z}
          xLabel="rank in embedding"
          yLabel="rank in data"
          valueLabel="log₁₀(1 + pairs)"
          scale="sequential"
          height={340}
        />
        <XYChart
          height={340}
          xLabel="K"
          yLabel="quality"
          yRange={[0, 1]}
          handles={[{ kind: 'x', at: k.value, onDrag: k.set, label: 'K' }]}
          series={[
            { name: 'trustworthiness', type: 'line', x: result.ks, y: result.curves.map((c) => c.trust), slot: 0 },
            { name: 'continuity', type: 'line', x: result.ks, y: result.curves.map((c) => c.cont), slot: 1 },
            { name: 'Q_NX (neighbours kept)', type: 'line', x: result.ks, y: result.curves.map((c) => c.qnx), slot: 2 },
          ]}
        />
      </div>
    </Interactive>
  )
}

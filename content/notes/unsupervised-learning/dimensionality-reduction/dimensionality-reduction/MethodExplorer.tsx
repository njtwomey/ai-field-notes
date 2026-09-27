import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import {
  classicalMds,
  diffusionMap,
  distances,
  isomap,
  kernelPca,
  kthDistance,
  laplacianEigenmaps,
  manifold,
  pca,
  ranks,
  trustworthiness,
  tsneEmbedding,
  umap,
  type DatasetId,
  type Rows,
} from '../../_shared/manifold'

const N = 240
/** Neighbourhood size at which trustworthiness and continuity are scored, fixed so that methods compare fairly. */
const SCORE_K = 10

const DATASETS = [
  { value: 'swiss-roll', label: 'Swiss roll' },
  { value: 's-curve', label: 'S-curve' },
  { value: 'moons', label: 'two moons' },
  { value: 'blobs', label: 'blobs' },
  { value: 'circle', label: 'noisy circle' },
] as const

const METHODS = [
  { value: 'pca', label: 'PCA' },
  { value: 'mds', label: 'classical MDS' },
  { value: 'kpca', label: 'kernel PCA' },
  { value: 'isomap', label: 'Isomap' },
  { value: 'le', label: 'Laplacian eigenmaps' },
  { value: 'dm', label: 'diffusion map' },
  { value: 'tsne', label: 't-SNE' },
  { value: 'umap', label: 'UMAP' },
] as const

type Method = (typeof METHODS)[number]['value']

/** Uses of k by method: the neighbourhood graph, or the kernel width for kernel PCA. t-SNE uses perplexity 30. */
const USES_K: Record<Method, boolean> = {
  pca: false,
  mds: false,
  kpca: true,
  isomap: true,
  le: true,
  dm: true,
  tsne: false,
  umap: true,
}

/** Points in three bands of the intrinsic coordinate, or by cluster label, as one grouped scatter series. */
function coloured(xy: Rows, t: number[], labelled: boolean, name: string): XYSeries {
  let group: number[]
  let groupNames: string[]
  if (labelled) {
    group = t
    groupNames = [...new Set(t)].sort().map((g) => `cluster ${g + 1}`)
  } else {
    const sorted = [...t].sort((a, b) => a - b)
    const cut1 = sorted[Math.floor(t.length / 3)]
    const cut2 = sorted[Math.floor((2 * t.length) / 3)]
    group = t.map((v) => (v < cut1 ? 0 : v < cut2 ? 1 : 2))
    groupNames = ['first third', 'middle third', 'last third']
  }
  return { name, type: 'scatter', x: xy.map((p) => p[0]), y: xy.map((p) => p[1]), group, groupNames }
}

export function MethodExplorer() {
  const [dataset, setDataset] = useState<DatasetId>('swiss-roll')
  const [method, setMethod] = useState<Method>('isomap')
  const [k, setK] = useState(6)
  const data = useMemo(() => {
    const m = manifold(dataset, N)
    const d = distances(m.x)
    return { ...m, d, r: ranks(d) }
  }, [dataset])
  const kUsed = USES_K[method] ? k : 0
  const fit = useMemo(() => {
    let connected: boolean | undefined
    let y: Rows
    switch (method) {
      case 'pca':
        y = pca(data.x)
        break
      case 'mds':
        y = classicalMds(data.d)
        break
      case 'kpca':
        y = kernelPca(data.d, 2 * kthDistance(data.d, kUsed))
        break
      case 'isomap': {
        const out = isomap(data.d, kUsed)
        y = out.y
        connected = out.connected
        break
      }
      case 'le':
        y = laplacianEigenmaps(data.d, kUsed)
        break
      case 'dm':
        y = diffusionMap(data.d, kUsed)
        break
      case 'tsne':
        y = tsneEmbedding(data.x, 30)
        break
      case 'umap':
        y = umap(data.d, kUsed)
        break
    }
    return { y, connected, ...trustworthiness(data.r, ranks(distances(y)), SCORE_K) }
  }, [data, method, kUsed])
  const [a, b] = data.view
  const view = data.x.map((p) => [p[a], p[b]])

  return (
    <Interactive
      title="Eight methods on five data sets"
      caption={
        <>
          240 points in three dimensions. Left: the data seen along two of its axes. Right: the two-dimensional
          embedding. Colour marks the intrinsic coordinate (position along the roll, the S or the circle, in thirds) or
          the cluster. On the Swiss roll, PCA and classical MDS (identical here) flatten the layers onto each other;
          Isomap unrolls the sheet for k from 4 to 7 and folds it once edges jump between layers from k = 8; Laplacian
          eigenmaps and diffusion maps order the points along the roll but bend the sheet into an arc. t-SNE and UMAP
          keep neighbourhoods but not the shape. Diffusion maps here use the same k-nearest-neighbour graph; kernel PCA
          uses a Gaussian kernel of width twice the mean distance to the k-th neighbour.
        </>
      }
      controls={
        <>
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice label="data" value={dataset} onChange={setDataset} options={DATASETS} />
          </div>
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamChoice label="method" value={method} onChange={setMethod} options={METHODS} />
          </div>
          <ParamSlider label="k (neighbours)" value={k} onChange={setK} min={4} max={20} step={1} />
        </>
      }
      readout={
        <>
          <Readout label={`trustworthiness (k = ${SCORE_K})`} value={formatNumber(fit.trust)} />
          <Readout label={`continuity (k = ${SCORE_K})`} value={formatNumber(fit.cont)} />
          <Readout label="uses k" value={USES_K[method] ? 'yes' : 'no'} />
          {fit.connected !== undefined && (
            <Readout label="graph" value={fit.connected ? 'connected' : 'disconnected'} />
          )}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel={`x${'₁₂₃'[a]}`}
          yLabel={`x${'₁₂₃'[b]}`}
          series={[coloured(view, data.t, data.labelled, 'data')]}
        />
        <XYChart
          height={320}
          xLabel="embedding 1"
          yLabel="embedding 2"
          series={[coloured(fit.y, data.t, data.labelled, 'embedding')]}
        />
      </div>
    </Interactive>
  )
}

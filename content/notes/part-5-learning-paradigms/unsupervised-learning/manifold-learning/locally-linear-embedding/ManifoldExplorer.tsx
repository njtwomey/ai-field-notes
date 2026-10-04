import { useMemo, useState } from 'react'
import { dataset } from 'aifn/learning/estimators'
import { swissRoll } from 'aifn-applied/data/synthetic'
import { classicalMds, kernelPca, metricMds, pca } from 'aifn-applied/unsupervised/embedding/linear'
import { isomap, laplacianEigenmaps, locallyLinearEmbedding } from 'aifn-applied/unsupervised/embedding/manifold'
import { rbf } from 'aifn/learning/kernels'
import { pairwiseDistances } from 'aifn/numerics/linalg'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Plot,
  Points,
  Readout,
  useAxis,
  useScaleColor,
} from 'aifn-render'

const METHOD_OPTIONS = [
  { value: 'lle', label: 'LLE (Locally Linear Embedding)' },
  { value: 'isomap', label: 'Isomap' },
  { value: 'eigenmaps', label: 'Laplacian eigenmaps' },
  { value: 'pca', label: 'PCA' },
  { value: 'mds', label: 'classical MDS' },
  { value: 'smacof', label: 'metric MDS (SMACOF)' },
  { value: 'kpca', label: 'kernel PCA (RBF, ℓ = 4)' },
] as const

type MethodKey = (typeof METHOD_OPTIONS)[number]['value']

export function ManifoldExplorer() {
  const [method, setMethod] = useState<MethodKey>('lle')
  const [kNeighbours, setKNeighbours] = useState(10)
  const [nPoints, setNPoints] = useState(300)

  const data = useMemo(
    () => swissRoll(stream('content/embed/swiss-roll'), { n: nPoints, noise: 0.1, height: 8 }),
    [nPoints],
  )

  const embedded = useMemo(() => {
    const x = data.x
    let emb: Tensor
    switch (method) {
      case 'pca':
        emb = pca({ components: 2 }).fit(dataset(x)).transform(x)
        break
      case 'mds':
        emb = classicalMds(pairwiseDistances(x)).embedding
        break
      case 'smacof':
        emb = metricMds({ maxSteps: 80 }).fit(dataset(x)).embedding
        break
      case 'kpca':
        emb = kernelPca({ kernel: rbf({ lengthscale: 4 }) }).fit(dataset(x)).embedding
        break
      case 'isomap':
        emb = isomap({ neighbours: kNeighbours }).fit(dataset(x)).embedding
        break
      case 'eigenmaps':
        emb = laplacianEigenmaps({ neighbours: kNeighbours }).fit(dataset(x)).embedding
        break
      case 'lle':
      default:
        emb = locallyLinearEmbedding({ neighbours: kNeighbours }).fit(dataset(x)).embedding
        break
    }
    const rows = toRows(emb)
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [method, kNeighbours, data.x])

  const t = useMemo(() => toFlat(data.t!), [data])
  const colour = useScaleColor('sequential')
  const [lo, hi] = useMemo(() => [Math.min(...t), Math.max(...t)], [t])
  const colours = useMemo(() => t.map((v) => colour((v - lo) / Math.max(1e-6, hi - lo))), [t, colour, lo, hi])

  const z0 = useAxis({ label: 'z₀', hold: 'initial', key: method })
  const z1 = useAxis({ label: 'z₁', hold: 'initial', key: method, equal: z0 })

  const isNeighbourMethod = method === 'lle' || method === 'isomap' || method === 'eigenmaps'

  return (
    <Figure
      title="Unrolling the Swiss roll: linear vs manifold embeddings"
      purpose="Every method maps the 3-D Swiss roll to the 2-D plane. Point colour represents intrinsic position along the roll spiral. A successful manifold embedding unwraps the spiral into a continuous strip, preserving the rainbow ordering."
      defaultSize="L"
      controls={
        <ControlGroup>
          <Select
            label="Method"
            options={METHOD_OPTIONS}
            value={method}
            onChange={(v) => setMethod(v as MethodKey)}
          />
          {isNeighbourMethod && (
            <NumberSelector
              label="Neighbours k"
              value={kNeighbours}
              onChange={setKNeighbours}
              min={4}
              max={30}
              step={1}
              suggestions={[6, 8, 10, 15, 20]}
            />
          )}
          <NumberSelector
            label="Points n"
            value={nPoints}
            onChange={setNPoints}
            min={100}
            max={600}
            step={50}
            suggestions={[150, 300, 450]}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="Method" value={METHOD_OPTIONS.find((m) => m.value === method)?.label ?? method} />
          <Readout label="Sample size" value={nPoints} />
          {isNeighbourMethod && <Readout label="k (neighbours)" value={kNeighbours} />}
        </>
      }
      caption="Linear methods (PCA, classical and metric MDS on Euclidean distances) project opposite layers of the spiral onto each other and mix colours. Isomap uses shortest graph paths to unroll the sheet into a rectangular sheet; LLE and Laplacian eigenmaps reconstruct local patches, unfolding the manifold without requiring global Euclidean distance preservation."
    >
      <Plot x={z0} y={z1}>
        <Points name="points" x={embedded.x} y={embedded.y} colors={colours} />
      </Plot>
    </Figure>
  )
}

import { useMemo, useState } from 'react'
import { Figure, ControlGroup, Select, NumberSelector, Plots, Plot, Points, Readout, useAxis } from 'aifn-render'
import { circles, moons } from 'aifn-methods/data/synthetic'
import { kmeans, spectralClustering } from 'aifn-methods/unsupervised/clustering'
import { dataset } from 'aifn-compute/learning/estimators'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'

type Shape = 'circles' | 'moons'
type ViewMode = 'compare' | 'embedding'

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

export function SpectralExplorer() {
  const [shape, setShape] = useState<Shape>('circles')
  const [lengthscale, setLengthscale] = useState(0.15)
  const [view, setView] = useState<ViewMode>('compare')

  const data = useMemo(() => {
    if (shape === 'circles') {
      return circles(stream('spectral/circles'), { n: 200, noise: 0.05, factor: 0.5 })
    }
    return moons(stream('spectral/moons'), { n: 200, noise: 0.08 })
  }, [shape])

  const cols = useMemo(() => columns(data.x), [data])

  const km = useMemo(
    () => kmeans({ k: 2, restarts: 5 }).fit(dataset(data.x), { stream: stream('spectral/km') }),
    [data],
  )

  const sc = useMemo(
    () =>
      spectralClustering({
        k: 2,
        affinity: { kind: 'rbf', lengthscale },
        normaliseRows: true,
      }).fit(dataset(data.x), { stream: stream('spectral/sc') }),
    [data, lengthscale],
  )

  const kmLabels = useMemo(() => toFlat(km.decide(data.x)), [km, data])
  const scLabels = useMemo(() => toFlat(sc.labels), [sc])

  const embedding = useMemo(() => {
    const rows = toRows(sc.embedding)
    return { u0: rows.map((r) => r[0]), u1: rows.map((r) => r[1]) }
  }, [sc])

  const eigenvalues = useMemo(() => toFlat(sc.eigenvalues), [sc])

  // Axis hooks for 2D data
  const ax = useAxis({ label: view === 'compare' ? 'x₀ (k-means)' : 'x₀ (data space)' })
  const ay = useAxis({ label: 'x₁', equal: ax })

  const bx = useAxis({
    label: view === 'compare' ? 'x₀ (spectral)' : 'u₁ (eigenvector 1)',
    ...(view === 'embedding' ? { range: [-1.2, 1.2] } : {}),
  })
  const by = useAxis({
    label: view === 'compare' ? 'x₁' : 'u₂ (eigenvector 2)',
    equal: bx,
    ...(view === 'embedding' ? { range: [-1.2, 1.2] } : {}),
  })

  return (
    <Figure
      title="Spectral clustering against k-means"
      purpose="k-means cuts the raw feature space into convex Voronoi cells, making it impossible to separate concentric rings or interleaved manifolds; spectral clustering projects onto the bottom eigenvectors of the normalised graph Laplacian, where non-linear manifolds unroll into easily separable clusters."
      defaultSize="L"
      controls={
        <ControlGroup title="Clustering configuration">
          <Select
            label="dataset"
            value={shape}
            onChange={(v) => setShape(v as Shape)}
            options={[
              { value: 'circles', label: 'concentric circles' },
              { value: 'moons', label: 'interlocking moons' },
            ]}
          />
          <NumberSelector
            label="affinity lengthscale ℓ"
            value={lengthscale}
            onChange={(v) => setLengthscale(Math.max(0.02, Math.min(1.0, v)))}
            min={0.02}
            max={1.0}
            step={0.01}
            suggestions={[0.05, 0.1, 0.15, 0.25, 0.5]}
          />
          <Select
            label="view mode"
            value={view}
            onChange={(v) => setView(v as ViewMode)}
            options={[
              { value: 'compare', label: 'k-means vs Spectral in 2D' },
              { value: 'embedding', label: 'Spectral clusters & 2D Laplacian embedding' },
            ]}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout
            label="top eigenvalues"
            value={eigenvalues
              .slice(0, 3)
              .map((v) => v.toFixed(4))
              .join(', ')}
          />
          <Readout label="connected components" value={sc.components} />
          <Readout
            label="eigengap (λ₂ - λ₁)"
            value={eigenvalues.length >= 2 ? (eigenvalues[1] - eigenvalues[0]).toFixed(4) : '—'}
          />
        </>
      }
      caption="Left plot shows either raw k-means partition (which divides the plane with a linear cut) or the spectral result. Right plot contrasts the spectral clustering in 2D or reveals the normalised Laplacian eigenvector coordinates (u₁, u₂), where points on separate manifolds collapse onto distinct orthogonal poles on the unit circle."
    >
      <Plots cols={2}>
        <Plot x={ax} y={ay} legend={false}>
          <Points
            name="left-points"
            x={cols.x0}
            y={cols.x1}
            group={view === 'compare' ? kmLabels : scLabels}
            groupNames={['cluster 1', 'cluster 2']}
          />
        </Plot>
        <Plot x={bx} y={by} legend={false}>
          <Points
            name="right-points"
            x={view === 'compare' ? cols.x0 : embedding.u0}
            y={view === 'compare' ? cols.x1 : embedding.u1}
            group={scLabels}
            groupNames={['cluster 1', 'cluster 2']}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}

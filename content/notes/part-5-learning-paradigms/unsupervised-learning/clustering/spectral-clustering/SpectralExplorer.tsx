import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber } from 'aifn-render'
import { moons, rings } from '../../_shared/datasets'
import { kmeans, spectralTwo } from './spectral'

type Shape = 'rings' | 'moons'
const SHAPES = [
  { value: 'rings', label: 'two rings' },
  { value: 'moons', label: 'two moons' },
] as const satisfies readonly { value: Shape; label: string }[]

/** Fraction of points whose cluster matches the generating label, up to swapping the two cluster names. */
const agreement = (labels: number[], truth: number[]) => {
  const same = labels.filter((l, i) => l === truth[i]).length / labels.length
  return Math.max(same, 1 - same)
}

export function SpectralExplorer() {
  const [shape, setShape] = useState<Shape>('rings')
  const [sigma, setSigma] = useState(0.1)
  const data = useMemo(() => (shape === 'rings' ? rings(120, 0.4, 0.05, 3) : moons(120, 0.07, 7)), [shape])
  const fit = useMemo(() => spectralTwo(data.points, sigma), [data, sigma])
  const kmeansAgreement = useMemo(() => agreement(kmeans(data.points, 2), data.labels), [data])
  const [l1, l2, l3] = fit.laplacianEigenvalues
  const names = ['cluster 1', 'cluster 2']

  return (
    <Interactive
      title="Spectral clustering and its embedding"
      caption="Left: the points coloured by spectral clustering with a Gaussian similarity of width σ. Right: each point's row of the two leading eigenvectors, scaled to unit length; k-means runs on these. For σ between about 0.05 and 0.15 the two groups separate in the embedding and the clustering is exact. A large σ links the groups, the small eigenvalues stop being near zero, and the clustering degrades toward what k-means would do."
      controls={
        <>
          <ParamChoice label="data" value={shape} onChange={setShape} options={SHAPES} />
          <ParamSlider
            label="σ (similarity width)"
            value={sigma}
            onChange={setSigma}
            min={0.03}
            max={0.6}
            step={0.01}
          />
        </>
      }
      readout={
        <>
          <Readout
            label="Laplacian eigenvalues λ₁, λ₂, λ₃"
            value={[l1, l2, l3].map((v) => formatNumber(Math.max(v, 0))).join(', ')}
          />
          <Readout
            label="spectral agrees with generating labels"
            value={`${(100 * agreement(fit.labels, data.labels)).toFixed(0)}%`}
          />
          <Readout label="k-means on raw points agrees" value={`${(100 * kmeansAgreement).toFixed(0)}%`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          equalAspect
          xRange={shape === 'rings' ? [-1.3, 1.3] : [-1.3, 2.3]}
          yRange={shape === 'rings' ? [-1.3, 1.3] : [-0.9, 1.3]}
          xLabel="x₁"
          yLabel="x₂"
          series={[
            {
              name: 'points',
              type: 'scatter',
              x: data.points.map((p) => p[0]),
              y: data.points.map((p) => p[1]),
              group: fit.labels,
              groupNames: names,
            },
          ]}
        />
        <XYChart
          equalAspect
          xRange={[-1.2, 1.2]}
          yRange={[-1.2, 1.2]}
          xLabel="eigenvector 1 (normalised)"
          yLabel="eigenvector 2 (normalised)"
          series={[
            {
              name: 'embedding',
              type: 'scatter',
              x: fit.embedding.map((p) => p[0]),
              y: fit.embedding.map((p) => p[1]),
              group: fit.labels,
              groupNames: names,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}

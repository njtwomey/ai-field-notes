import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Handle } from '@/components/viz'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import type { Point } from '@/lib/math/cluster'
import { converge } from './lloyd'

const K_MAX = 8

/**
 * Inertia against k on the blobs from python/mlc/figures/kmeans.py, each k the best of several k-means++ runs, with
 * the clustering at the chosen k beside it.
 */
export function ElbowCurve() {
  const { data } = useFigure<PointCloud2d>('k-means/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const k = useParam(3, { min: 1, max: K_MAX, step: 1 })
  const [restarts, setRestarts] = useState(5)

  const fits = useMemo(() => {
    if (!points.length) return []
    return Array.from({ length: K_MAX }, (_, i) => {
      const runs = Array.from({ length: restarts }, (_, seed) => converge(points, i + 1, 'kmeans++', seed))
      return runs.reduce((best, r) => (r.inertia < best.inertia ? r : best))
    })
  }, [points, restarts])

  if (!data || !fits.length) return null
  const ks = fits.map((_, i) => i + 1)
  const chosen = fits[k.value - 1]
  const handles: Handle[] = [{ kind: 'x', at: k.value, label: 'k', onDrag: (x) => k.set(Math.round(x)) }]

  return (
    <Interactive
      title="The elbow curve"
      caption="Inertia always falls as k grows, so its minimum is useless. The elbow method looks instead for the k after which the fall flattens: here k = 3, the number of blobs. Drag the line labelled k, or use the slider, to see the clustering at each k. Each k is the best of several k-means++ runs."
      controls={
        <>
          <ParamSlider label="k" param={k} />
          <ParamSlider label="restarts per k" value={restarts} onChange={setRestarts} min={1} max={10} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="inertia" value={formatNumber(chosen.inertia)} />
          <Readout
            label="drop from k − 1"
            value={k.value > 1 ? formatNumber(fits[k.value - 2].inertia - chosen.inertia) : '—'}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="k"
          yLabel="inertia"
          yRange={[0, undefined]}
          handles={handles}
          series={[
            { name: 'inertia', type: 'line', x: ks, y: fits.map((f) => f.inertia) },
            { name: 'chosen k', type: 'scatter', x: [k.value], y: [chosen.inertia], emphasis: true },
          ]}
        />
        <XYChart
          height={320}
          xLabel="x₁"
          yLabel="x₂"
          series={[
            {
              name: 'points',
              type: 'scatter',
              x: data.x,
              y: data.y,
              group: chosen.labels,
              groupNames: chosen.centroids.map((_, j) => `cluster ${j + 1}`),
            },
            {
              name: 'centroids',
              type: 'scatter',
              x: chosen.centroids.map((c) => c[0]),
              y: chosen.centroids.map((c) => c[1]),
              emphasis: true,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}

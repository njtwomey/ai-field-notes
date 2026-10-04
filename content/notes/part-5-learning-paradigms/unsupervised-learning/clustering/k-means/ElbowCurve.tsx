import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import type { Vec2 as Point } from 'aifn/numerics/linalg'
import { converge } from './lloyd'

const K_MAX = 8

/**
 * Inertia against k on the blobs from python/mlc/figures/kmeans.py, each k the best of several k-means++ runs, with
 * the clustering at the chosen k beside it.
 */
export function ElbowCurve() {
  const { data } = useFigure<PointCloud2d>('k-means/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const state = useFigureState({
    k: int(3, { min: 1, max: K_MAX, step: 1, label: 'k' }),
    restarts: int(5, { min: 1, max: 10, step: 1, label: 'restarts per k' }),
  })

  const fits = useMemo(() => {
    if (!points.length) return []
    return Array.from({ length: K_MAX }, (_, i) => {
      const runs = Array.from({ length: state.restarts }, (_, seed) => converge(points, i + 1, 'kmeans++', seed))
      return runs.reduce((best, r) => (r.inertia < best.inertia ? r : best))
    })
  }, [points, state.restarts])

  const xAxis = useAxis({ label: 'k', hold: 'union' })
  const yAxis = useAxis({ label: 'inertia', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'x₁', hold: 'union' })
  const yAxis2 = useAxis({ label: 'x₂', hold: 'union' })
  if (!data || !fits.length) return null
  const ks = fits.map((_, i) => i + 1)
  const chosen = fits[state.k - 1]

  return (
    <Figure
      title="The elbow curve"
      state={state}
      caption="Inertia always falls as k grows, so its minimum is useless. The elbow method looks instead for the k after which the fall flattens: here k = 3, the number of blobs. Drag the line labelled k, or use the slider, to see the clustering at each k. Each k is the best of several k-means++ runs."

      readouts={
        <>
          <Readout label="inertia" value={formatNumber(chosen.inertia)} />
          <Readout
            label="drop from k − 1"
            value={state.k > 1 ? formatNumber(fits[state.k - 2].inertia - chosen.inertia) : '—'}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve name="inertia" x={ks} y={fits.map((f) => f.inertia)} />
          <Points name="chosen k" x={[state.k]} y={[chosen.inertia]} emphasis />
          <Handle kind="x" at={state.k} label="k" onDrag={(x) => state.set('k', Math.round(x))} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Points
            name="points"
            x={data.x}
            y={data.y}
            group={chosen.labels}
            groupNames={chosen.centroids.map((_, j) => `cluster ${j + 1}`)}
          />
          <Points
            name="centroids"
            x={chosen.centroids.map((c) => c[0])}
            y={chosen.centroids.map((c) => c[1])}
            emphasis
          />
        </Plot>
      </div>
    </Figure>
  )
}

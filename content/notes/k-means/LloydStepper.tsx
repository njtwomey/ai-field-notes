import { useEffect, useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, StepControls, XYChart, formatNumber } from '@/components/viz'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { CENTRE_INIT_OPTIONS, d2, initialCentres, type CentreInit, type Point } from '@/lib/math/cluster'

type State = { centroids: Point[]; labels: number[]; inertia: number; iteration: number; done: boolean }

function assign(points: Point[], centroids: Point[]) {
  const labels = points.map((p) => centroids.reduce((best, c, j) => (d2(p, c) < d2(p, centroids[best]) ? j : best), 0))
  const inertia = points.reduce((s, p, i) => s + d2(p, centroids[labels[i]]), 0)
  return { labels, inertia }
}

function step(points: Point[], s: State): State {
  const k = s.centroids.length
  const centroids = s.centroids.map((c, j) => {
    const members = points.filter((_, i) => s.labels[i] === j)
    if (!members.length) return c
    return [
      members.reduce((a, p) => a + p[0], 0) / members.length,
      members.reduce((a, p) => a + p[1], 0) / members.length,
    ] as Point
  })
  const moved = centroids.some((c, j) => d2(c, s.centroids[j]) > 1e-12)
  const { labels, inertia } = assign(points, centroids)
  return { centroids, labels, inertia, iteration: s.iteration + 1, done: !moved || k === 0 }
}

/** Lloyd's algorithm one step at a time on data from python/mlc/figures/kmeans.py. */
export function LloydStepper() {
  const { data } = useFigure<PointCloud2d>('k-means/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const [k, setK] = useState(3)
  const [init, setInit] = useState<CentreInit>('random')
  const [seed, setSeed] = useState(3)
  const [state, setState] = useState<State>()

  const reset = () => {
    if (!points.length) return
    const centroids = initialCentres(points, k, init, seed)
    setState({ centroids, ...assign(points, centroids), iteration: 0, done: false })
  }
  // Re-initialise whenever the data or a setting changes.
  useEffect(reset, [points, k, init, seed]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!data || !state) return null
  const runToEnd = () => {
    let s = state
    for (let i = 0; i < 100 && !s.done; i++) s = step(points, s)
    setState(s)
  }

  return (
    <Interactive
      title="Lloyd's algorithm, step by step"
      caption="Each step moves every centroid to the mean of its points, then reassigns points to the nearest centroid. Inertia never increases. Try random initialisation with different seeds: some runs converge to a worse split."
      controls={
        <>
          <ParamSlider label="k" value={k} onChange={setK} min={1} max={6} step={1} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={20} step={1} />
          <ParamChoice label="initialisation" value={init} onChange={setInit} options={CENTRE_INIT_OPTIONS} />
          <StepControls
            onStep={() => setState(step(points, state))}
            onRun={runToEnd}
            onReset={reset}
            done={state.done}
          />
        </>
      }
      readout={
        <>
          <Readout label="iteration" value={state.iteration} />
          <Readout label="inertia" value={formatNumber(state.inertia)} />
          <Readout label="status" value={state.done ? 'converged' : 'running'} />
        </>
      }
    >
      <XYChart
        height={380}
        xLabel="x₁"
        yLabel="x₂"
        series={[
          {
            name: 'points',
            type: 'scatter',
            x: data.x,
            y: data.y,
            group: state.labels,
            groupNames: state.centroids.map((_, j) => `cluster ${j + 1}`),
          },
          {
            name: 'centroids',
            type: 'scatter',
            x: state.centroids.map((c) => c[0]),
            y: state.centroids.map((c) => c[1]),
            emphasis: true,
          },
        ]}
      />
    </Interactive>
  )
}

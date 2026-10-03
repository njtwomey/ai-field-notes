import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  StepControls,
  XYChart,
  formatNumber,
  type Handle,
} from 'aifn-render'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { useDerivedState } from '@/lib/use-derived-state'
import { CENTRE_INIT_OPTIONS, initialCentres, type CentreInit, type Point } from '@/lib/math/cluster'
import { assign, step, type State } from './lloyd'

/** Lloyd's algorithm one step at a time on data from python/mlc/figures/kmeans.py. */
export function LloydStepper() {
  const { data } = useFigure<PointCloud2d>('k-means/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const [k, setK] = useState(3)
  const [init, setInit] = useState<CentreInit>('random')
  const [seed, setSeed] = useState(3)
  // Re-initialised whenever the data or a setting changes.
  const initial = useMemo((): State | undefined => {
    if (!points.length) return undefined
    const centroids = initialCentres(points, k, init, seed)
    return { centroids, ...assign(points, centroids), iteration: 0, done: false }
  }, [points, k, init, seed])
  const [state, setState, reset] = useDerivedState(initial)

  if (!data || !state) return null
  // Dragging a centroid starts a new run from the edited centroids: points are reassigned and the count restarts,
  // since a hand-placed centroid can raise the inertia and the iterations before it no longer describe this run.
  const handles: Handle[] = state.centroids.map((c, j) => ({
    kind: 'point',
    at: c,
    label: `centroid ${j + 1}`,
    onDrag: (p) =>
      setState((s) => {
        if (!s) return s
        const centroids = s.centroids.map((old, i) => (i === j ? p : old))
        return { centroids, ...assign(points, centroids), iteration: 0, done: false }
      }),
  }))
  const runToEnd = () => {
    let s = state
    for (let i = 0; i < 100 && !s.done; i++) s = step(points, s)
    setState(s)
  }

  return (
    <Interactive
      title="Lloyd's algorithm, step by step"
      caption="Each step moves every centroid to the mean of its points, then reassigns points to the nearest centroid. Inertia never increases. Try random initialisation with different seeds: some runs converge to a worse split. Drag a centroid to place it by hand; the run restarts from there."
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
        handles={handles}
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

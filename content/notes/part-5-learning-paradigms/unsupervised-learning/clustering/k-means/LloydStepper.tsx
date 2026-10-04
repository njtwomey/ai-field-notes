import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import type { Vec2 as Point } from 'aifn/numerics/linalg'
import { CENTRE_INIT_OPTIONS, initialCentres, type CentreInit } from '../_shared/centres'
import { assign, step, type State } from './lloyd'

/** Every iteration from the starting centroids to convergence (at most 100 steps). */
function trace(points: Point[], centroids: Point[]): State[] {
  let s: State = { centroids, ...assign(points, centroids), iteration: 0, done: false }
  const out = [s]
  for (let i = 0; i < 100 && !s.done; i++) out.push((s = step(points, s)))
  return out
}

/** Lloyd's algorithm one step at a time on data from python/mlc/figures/kmeans.py. */
export function LloydStepper() {
  const { data } = useFigure<PointCloud2d>('k-means/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const fig = useFigureState({
    k: int(3, { min: 1, max: 6, step: 1, label: 'k', suggestions: [2, 3, 4, 6] }),
    seed: int(3, { ge: 0, label: 'seed' }),
    init: choice<CentreInit>(CENTRE_INIT_OPTIONS, 'random', { label: 'initialisation' }),
  })
  const { k, seed, init } = fig
  // Hand-placed starting centroids and the player position, each stored with the settings it belongs to, so that a
  // new k, seed or initialisation restarts the walk-through from step 0.
  const settings = `${k}/${seed}/${init}`
  const [placed, setPlaced] = useState<{ settings: string; centroids: Point[] } | null>(null)
  const [pos, setPos] = useState({ settings, at: 0 })
  const start = useMemo(() => {
    if (placed?.settings === settings) return placed.centroids
    return points.length ? initialCentres(points, k, init, seed) : []
  }, [placed, settings, points, k, init, seed])
  const states = useMemo(() => (points.length ? trace(points, start) : []), [points, start])
  const at = pos.settings === settings ? Math.min(pos.at, states.length - 1) : 0
  const state = states[at]

  const xAxis = useAxis({ label: 'x₁', hold: 'union' })
  const yAxis = useAxis({ label: 'x₂', hold: 'union' })
  if (!data || !state) return null
  // Dragging a centroid starts a new run from the edited centroids: points are reassigned and the count restarts,
  // since a hand-placed centroid can raise the inertia and the iterations before it no longer describe this run.
  const handles: Handle[] = state.centroids.map((c, j) => ({
    kind: 'point',
    at: c,
    label: `centroid ${j + 1}`,
    onDrag: (p) => {
      setPlaced({ settings, centroids: state.centroids.map((old, i) => (i === j ? p : old)) })
      setPos({ settings, at: 0 })
    },
  }))

  return (
    <Figure
      title="Lloyd's algorithm, step by step"
      caption="Each step moves every centroid to the mean of its points, then reassigns points to the nearest centroid. Inertia never increases. Step through the iterations with the player. Try random initialisation with different seeds: some runs converge to a worse split. Drag a centroid to place it by hand; the run restarts from there."
      state={fig}
      controls={
        <Player value={at} onChange={(v) => setPos({ settings, at: v })} count={states.length} label="iteration" />
      }
      readouts={
        <>
          <Readout label="iteration" value={state.iteration} />
          <Readout label="inertia" value={formatNumber(state.inertia)} />
          <Readout label="status" value={state.done ? 'converged' : 'running'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        <Points
          name="points"
          x={data.x}
          y={data.y}
          group={state.labels}
          groupNames={state.centroids.map((_, j) => `cluster ${j + 1}`)}
        />
        <Points name="centroids" x={state.centroids.map((c) => c[0])} y={state.centroids.map((c) => c[1])} emphasis />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}

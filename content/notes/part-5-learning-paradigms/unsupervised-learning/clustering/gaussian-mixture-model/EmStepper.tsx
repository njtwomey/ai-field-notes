import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { PointCloud2d } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import type { Vec2 as Point } from 'aifn/numerics/linalg'
import { CENTRE_INIT_OPTIONS, type CentreInit } from '../_shared/centres'
import { ellipse, initialise, moveMean, seek, step, type EmState } from './em'

const MAX_RUN = 300

/** EM run from a starting state until it converges (at most MAX_RUN iterations); the cursor is left at the start. */
function run(points: Point[], start: EmState): EmState {
  let s = start
  for (let i = 0; i < MAX_RUN && !s.done; i++) s = step(points, s)
  return seek(points, s, 0)
}

/** EM for a diagonal Gaussian mixture, one iteration at a time, on data from python/mlc/figures/gmm.py. */
export function EmStepper() {
  const { data } = useFigure<PointCloud2d>('gaussian-mixture-model/blobs')
  const points = useMemo(() => (data ? data.x.map((x, i) => [x, data.y[i]] as Point) : []), [data])
  const fig = useFigureState({
    k: int(3, { min: 1, max: 6, step: 1, label: 'k', suggestions: [2, 3, 4, 6] }),
    seed: int(5, { ge: 0, label: 'seed' }),
    init: choice<CentreInit>(CENTRE_INIT_OPTIONS, 'random', { label: 'initialisation' }),
  })
  const { k, seed, init } = fig
  // A mixture edited by hand and the player position, each stored with the settings it belongs to, so that a new k,
  // seed or initialisation restarts the walk-through from iteration 0.
  const settings = `${k}/${seed}/${init}`
  const [edited, setEdited] = useState<{ settings: string; start: EmState } | null>(null)
  const [pos, setPos] = useState({ settings, at: 0 })
  const full = useMemo((): EmState | undefined => {
    if (!points.length) return undefined
    const start = edited?.settings === settings ? edited.start : initialise(points, k, init, seed)
    return run(points, start)
  }, [points, edited, settings, k, init, seed])
  const last = full ? full.history.length - 1 : 0
  const at = pos.settings === settings ? Math.min(pos.at, last) : 0
  const state = useMemo(() => (full ? seek(points, full, at) : undefined), [points, full, at])

  const series = useMemo((): SeriesSpec[] => {
    if (!data || !state) return []
    const mixture = state.mixtures[state.cursor]
    const names = mixture.means.map((_, j) => `component ${j + 1}`)
    // Each point is coloured by its most likely component.
    const labels = state.responsibilities.map((r) => r.indexOf(Math.max(...r)))
    const ellipses = mixture.means.flatMap((_, j) =>
      [1, 2].map((radius): SeriesSpec => ({
        name: names[j],
        type: 'line',
        ...ellipse(mixture, j, radius),
        slot: j,
        dashed: radius === 2,
      })),
    )
    return [
      { name: 'points', type: 'scatter', x: data.x, y: data.y, group: labels, groupNames: names },
      ...ellipses,
      {
        name: 'means',
        type: 'scatter',
        x: mixture.means.map((m) => m[0]),
        y: mixture.means.map((m) => m[1]),
        emphasis: true,
      },
    ]
  }, [data, state])

  const curve = useMemo(
    (): SeriesSpec[] =>
      state
        ? [
            { name: 'log-likelihood', type: 'line', x: state.history.map((_, i) => i), y: state.history },
            // Marks the iteration on display.
            { name: 'shown', type: 'scatter', x: [state.cursor], y: [state.history[state.cursor]], emphasis: true },
          ]
        : [],
    [state],
  )

  const xAxis = useAxis({ label: 'x₁', range: [-9, 7] })
  const yAxis = useAxis({ label: 'x₂', range: [-3, 5], equal: xAxis })
  const xAxis2 = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis2 = useAxis({ label: 'mean log-likelihood', hold: 'union' })
  if (!data || !state) return null
  const meanHandles: Handle[] = state.mixtures[state.cursor].means.map((m, j) => ({
    kind: 'point',
    at: m,
    label: `mean ${j + 1}`,
    onDrag: (p) => {
      setEdited({ settings, start: moveMean(points, state, j, p) })
      setPos({ settings, at: 0 })
    },
  }))
  // Scrubbing the curve shows any iteration, as the player does; the position rounds to the nearest one.

  return (
    <Figure
      title="Expectation-maximisation, step by step"
      caption="Each step re-estimates every component from all points, weighted by how likely each point is to belong to it, then recomputes those probabilities. Points take the colour of their most likely component. Solid ellipses are one standard deviation from the mean; dashed ones are two. The log-likelihood never decreases. With random initialisation, seed 5 converges to a worse optimum and seed 3 stalls on a plateau before escaping. k-means++ reaches the better fit for every seed from 0 to 20. Step through the iterations with the player, or drag along the log-likelihood curve. Drag a mean to move it by hand; the run restarts from the edited mixture."
      state={fig}
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={last + 1} label="iteration" />}
      readouts={
        <>
          <Readout label="iteration" value={`${state.cursor} of ${last}`} />
          <Readout label="mean log-likelihood" value={formatNumber(state.history[state.cursor])} />
          <Readout label="status" value={state.cursor < last ? 'running' : state.done ? 'converged' : 'stopped'} />
          <Readout label="weights" value={state.mixtures[state.cursor].weights.map((w) => w.toFixed(2)).join(' · ')} />
        </>
      }
    >
      <div className="grid items-center gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          {(meanHandles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          {seriesLayers(curve)}
          <Handle
            kind="x"
            at={state.cursor}
            label="shown"
            onDrag={(x) => setPos({ settings, at: Math.max(0, Math.min(last, Math.round(x))) })}
          />
        </Plot>
      </div>
    </Figure>
  )
}

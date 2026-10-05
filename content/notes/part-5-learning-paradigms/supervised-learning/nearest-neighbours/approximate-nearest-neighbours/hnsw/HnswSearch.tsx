import { useMemo } from 'react'
import {
  Curve,
  Figure,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { useDerivedState } from '@/lib/use-derived-state'
import { nearest, type P } from '../_shared/geometry'
import { build, search } from './hnsw'
import { stream, uniform } from 'aifn-compute/foundation/random'

const N = 180
const M = 4
const EF_CONSTRUCTION = 24
const RANGE: [number, number] = [-4.2, 4.2]

/** HNSW on 180 random points in the plane: the layers, and a query's greedy descent through them. */
export function HnswSearch() {
  const state = useFigureState({
    ef: int(4, { min: 1, max: 24, step: 1, suggestions: [1, 4, 8, 24], label: 'beam width on layer 0, ef' }),
    qx: slider(RANGE[0], RANGE[1], 2.3, { step: 0.05, onChart: true }),
    qy: slider(RANGE[0], RANGE[1], -1.9, { step: 0.05, onChart: true }),
  })
  const query: P = useMemo(() => [state.qx, state.qy], [state.qx, state.qy])

  const graph = useMemo(() => {
    const g = stream(5)
    const points = Array.from({ length: N }, (): P => [8 * uniform(g) - 4, 8 * uniform(g) - 4])
    return build(points, M, EF_CONSTRUCTION, 17)
  }, [])
  const run = useMemo(() => search(graph, query, state.ef), [graph, query, state.ef])
  const last = run.steps.length - 1
  // The step shown resets to the end of the search whenever the query or ef changes.
  const initial = useMemo(() => ({ at: run.steps.length - 1 }), [run])
  const [cursor, setCursor] = useDerivedState(initial)
  const shown = cursor.at
  const step = run.steps[Math.min(shown, last)]
  const layer = step.layer
  const truth = nearest(graph.points, query)
  const foundTop = run.result[0]

  const edges = useMemo(
    (): Segment[] =>
      graph.links[layer].flatMap((ns, i) =>
        ns
          .filter((j) => j > i || !graph.links[layer][j].includes(i))
          .map((j) => ({ from: graph.points[i], to: graph.points[j] })),
      ),
    [graph, layer],
  )
  const onLayer = graph.points.flatMap((p, i) => (graph.level[i] >= layer ? [p] : []))
  const offLayer = graph.points.flatMap((p, i) => (graph.level[i] < layer ? [p] : []))
  const path = run.steps.slice(0, Math.min(shown, last) + 1).map((s) => graph.points[s.node])
  const evaluated = step.evaluated.map((i) => graph.points[i])

  const series = [
    {
      name: 'not on this layer',
      x: offLayer.map((p) => p[0]),
      y: offLayer.map((p) => p[1]),
      muted: true,
    },
    { name: `on layer ${layer}`, x: onLayer.map((p) => p[0]), y: onLayer.map((p) => p[1]), slot: 0 },
    {
      name: 'distance computed',
      x: evaluated.map((p) => p[0]),
      y: evaluated.map((p) => p[1]),
      slot: 1,
    },
    { name: 'true nearest', x: [graph.points[truth][0]], y: [graph.points[truth][1]], slot: 2 },
    { name: 'expanded nodes', x: path.map((p) => p[0]), y: path.map((p) => p[1]), emphasis: true },
  ] as const
  const layerCounts = graph.links.map((_, l) => graph.level.filter((v) => v >= l).length)

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="HNSW: greedy descent through the layers"
      state={state}
      caption={`${N} points inserted with M = ${M} links per node (${2 * M} on layer 0) and efConstruction = ${EF_CONSTRUCTION}. Layer sizes from the top: ${[...layerCounts].reverse().join(', ')}. The search starts at the entry point on the top layer, moves greedily to whichever neighbour is closest to the query, drops a layer when no neighbour is closer, and finishes with a beam of width ef on layer 0. Drag the query; step through the search with the player. Grey lines are the edges of the layer being searched.`}
      controls={
        <Player
          value={Math.min(shown, last)}
          onChange={(at) => setCursor({ at })}
          count={last + 1}
          label="step"
          format={(v) => String(v + 1)}
          startReason="a new query shows its whole search; step back through it"
        />
      }
      readouts={
        <>
          <Readout label="step" value={`${Math.min(shown, last) + 1} of ${last + 1}`} />
          <Readout label="layer" value={String(layer)} />
          <Readout label="distance computations" value={`${step.evaluated.length} of ${N}`} />
          <Readout label="full search finds the true nearest" value={foundTop === truth ? 'yes' : 'no'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Points {...series[0]} />
        <Points {...series[1]} />
        <Points {...series[2]} />
        <Points {...series[3]} />
        <Curve {...series[4]} />
        <Segments segments={edges} />
        <Handle {...state.handle(['qx', 'qy'], { label: 'query' })} />
      </Plot>
    </Figure>
  )
}

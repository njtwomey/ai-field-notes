import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  StepControls,
  XYChart,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { useDerivedState } from '@/lib/use-derived-state'
import { nearest, type P } from '../_shared/geometry'
import { build, search } from './hnsw'

const N = 180
const M = 4
const EF_CONSTRUCTION = 24
const RANGE: [number, number] = [-4.2, 4.2]

/** HNSW on 180 random points in the plane: the layers, and a query's greedy descent through them. */
export function HnswSearch() {
  const ef = useParam(4, { min: 1, max: 24, step: 1 })
  const [query, setQuery] = useState<P>([2.3, -1.9])

  const graph = useMemo(() => {
    const g = rng(5)
    const points = Array.from({ length: N }, (): P => [8 * g.uniform() - 4, 8 * g.uniform() - 4])
    return build(points, M, EF_CONSTRUCTION, 17)
  }, [])
  const run = useMemo(() => search(graph, query, ef.value), [graph, query, ef.value])
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

  const series: XYSeries[] = [
    {
      name: 'not on this layer',
      type: 'scatter',
      x: offLayer.map((p) => p[0]),
      y: offLayer.map((p) => p[1]),
      muted: true,
    },
    { name: `on layer ${layer}`, type: 'scatter', x: onLayer.map((p) => p[0]), y: onLayer.map((p) => p[1]), slot: 0 },
    {
      name: 'distance computed',
      type: 'scatter',
      x: evaluated.map((p) => p[0]),
      y: evaluated.map((p) => p[1]),
      slot: 1,
    },
    { name: 'true nearest', type: 'scatter', x: [graph.points[truth][0]], y: [graph.points[truth][1]], slot: 2 },
    { name: 'expanded nodes', type: 'line', x: path.map((p) => p[0]), y: path.map((p) => p[1]), emphasis: true },
  ]
  const handles: Handle[] = [{ kind: 'point', at: query, label: 'query', onDrag: (p) => setQuery(p) }]
  const layerCounts = graph.links.map((_, l) => graph.level.filter((v) => v >= l).length)

  return (
    <Interactive
      title="HNSW: greedy descent through the layers"
      caption={`${N} points inserted with M = ${M} links per node (${2 * M} on layer 0) and efConstruction = ${EF_CONSTRUCTION}. Layer sizes from the top: ${[...layerCounts].reverse().join(', ')}. The search starts at the entry point on the top layer, moves greedily to whichever neighbour is closest to the query, drops a layer when no neighbour is closer, and finishes with a beam of width ef on layer 0. Drag the query; step through the search with Reset and Step. Grey lines are the edges of the layer being searched.`}
      controls={
        <>
          <ParamSlider label="beam width on layer 0, ef" param={ef} format={(v) => String(v)} withArrows />
          <StepControls
            onStep={() => setCursor((c) => ({ at: Math.min(c.at + 1, last) }))}
            onRun={() => setCursor({ at: last })}
            onReset={() => setCursor({ at: 0 })}
            done={shown >= last}
          />
        </>
      }
      readout={
        <>
          <Readout label="step" value={`${Math.min(shown, last) + 1} of ${last + 1}`} />
          <Readout label="layer" value={String(layer)} />
          <Readout label="distance computations" value={`${step.evaluated.length} of ${N}`} />
          <Readout label="full search finds the true nearest" value={foundTop === truth ? 'yes' : 'no'} />
        </>
      }
    >
      <XYChart
        series={series}
        segments={edges}
        xRange={RANGE}
        yRange={RANGE}
        equalAspect
        xLabel="x₁"
        yLabel="x₂"
        handles={handles}
      />
    </Interactive>
  )
}

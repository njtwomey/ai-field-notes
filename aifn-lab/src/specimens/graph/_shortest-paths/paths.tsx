import { aStarSteps, dijkstraSteps, shortestPath, type DijkstraState } from 'aifn/graph/shortest-paths'
import { fromEdges, heapSorted, type Graph } from 'aifn/graph'
import { stream, uniform } from 'aifn/foundation/random'
import { trace, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { row, slider, useFigureState } from '@lab/state'
import { Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'
import { Sequence } from '../_shared/common'
import { stateAt } from '../_shared/data'

const W = 21
const H = 13
const SOURCE = 0
const TARGET = H * W - 1

/** A random maze: each cell a wall with the given probability, the corners kept open. */
function maze(seed: number, density: number): { walls: boolean[]; graph: Graph } {
  const s = stream(`graph-maze/${seed}`)
  const walls = Array.from({ length: H * W }, (_, v) => v !== SOURCE && v !== TARGET && uniform(s) < density)
  const edges: [number, number][] = []
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const v = r * W + c
      if (walls[v]) continue
      if (c + 1 < W && !walls[v + 1]) edges.push([v, v + 1])
      if (r + 1 < H && !walls[v + W]) edges.push([v, v + W])
    }
  return { walls, graph: fromEdges(H * W, edges, { directed: false }) }
}

/** The Manhattan distance to the target: admissible and consistent on a 4-connected grid with unit steps. */
const manhattan = (v: number) => Math.abs(Math.floor(v / W) - (H - 1)) + Math.abs((v % W) - (W - 1))

const cell = (v: number) => `(${Math.floor(v / W)},${v % W})`
const CATEGORIES = ['queued', 'expanded', 'path', 'current']
const XS = Array.from({ length: W }, (_, c) => c)
const YS = Array.from({ length: H }, (_, i) => i)

/** Live queue entries (not stale) in pop order. */
const liveQueue = (s: DijkstraState) =>
  heapSorted(s.queue).filter(
    (e) => !s.settled.data[e.value] && e.priority === s.distance.data[e.value] + s.heuristic.data[e.value],
  )

function grid(walls: boolean[], s: DijkstraState): number[][] {
  const code: number[] = walls.map((w) => (w ? -2 : -1))
  for (const e of liveQueue(s)) code[e.value] = 0
  s.settled.data.forEach((x, v) => x && (code[v] = 1))
  if (s.done) for (const v of shortestPath(s.predecessor, SOURCE, TARGET).data) code[v] = 2
  if (s.current >= 0 && !s.done) code[s.current] = 3
  // Row i of the heatmap is at y = i, drawn bottom-up: maze row 0 goes on top.
  return YS.map((i) => code.slice((H - 1 - i) * W, (H - i) * W))
}

function queueItems(s: DijkstraState): string[] {
  return liveQueue(s)
    .slice(0, 5)
    .map((e) => `${cell(e.value)} ${e.priority}`)
}

export function MazeSpecimen() {
  const state = useFigureState({
    maze: row('1 · maze', {
      seed: slider(1, 30, 3, { step: 1, label: 'maze' }),
      density: slider(0.1, 0.4, 0.28, { step: 0.02, label: 'wall density' }),
    }),
  })
  const { seed, density } = state.maze
  const { walls, graph } = useMemo(() => maze(seed, density), [seed, density])
  const traces = useMemo((): [Trace<DijkstraState>, Trace<DijkstraState>] => {
    const n = 4 * H * W
    return [
      trace(dijkstraSteps(graph, { source: SOURCE, target: TARGET }), undefined, n),
      trace(aStarSteps(graph, { source: SOURCE, target: TARGET, heuristic: manhattan }), undefined, n),
    ]
  }, [graph])
  const count = Math.max(traces[0].steps.length, traces[1].steps.length)
  const [step, setStep] = useState(0)
  const at = Math.min(step, count - 1)
  const d = stateAt(traces[0], at)
  const a = stateAt(traces[1], at)
  const zd = useMemo(() => grid(walls, d), [walls, d])
  const za = useMemo(() => grid(walls, a), [walls, a])
  const length = (s: DijkstraState) => (s.done ? s.distance.data[TARGET] : NaN)
  const x = useAxis({ label: 'column', zoom: false })
  const y = useAxis({ label: 'row', equal: x, zoom: false })
  const x2 = useAxis({ label: 'column', zoom: false })
  const y2 = useAxis({ label: 'row', equal: x2, zoom: false })

  return (
    <Figure
      title="Dijkstra against A* on a grid maze"
      purpose="A* adds an admissible estimate of the distance left to Dijkstra's priority, so it expands fewer cells and still finds a shortest path."
      defaultSize="M"
      hoverReadout={false}
      state={state}
      controls={
        <div className="col-span-full">
          <Player label="2 · expansion" value={at} onChange={setStep} count={count} />
        </div>
      }
      readouts={{
        Dijkstra: (
          <>
            <Readout label="expanded" value={d.expanded} />
            <Sequence label="queue" ends="least first" items={queueItems(d)} />
          </>
        ),
        'A*': (
          <>
            <Readout label="expanded" value={a.expanded} />
            <Sequence label="queue" ends="least first" items={queueItems(a)} />
          </>
        ),
        'shortest path': (
          <Readout
            label="length"
            value={
              !a.done
                ? '…'
                : Number.isFinite(length(a))
                  ? `${length(a)} steps (Dijkstra ${Number.isFinite(length(d)) ? length(d) : '…'})`
                  : 'no path'
            }
          />
        ),
      }}
      caption="From the top-left cell to the bottom-right, moving in four directions at unit cost. Each step pops the queued cell of least priority and expands it: Dijkstra's priority is the distance so far g, A*'s adds the Manhattan distance h to the goal, which never overestimates. Play to watch Dijkstra flood outwards while A* heads for the corner; both find a path of the same length (shown once done). The queues list their front entries as (row,col) priority."
    >
      <Plots cols={2}>
        <Plot x={x} y={y} title="Dijkstra: priority g" bare>
          <Raster x={XS} y={YS} z={zd} scale="categorical" categoryNames={CATEGORIES} valueLabel="Dijkstra" />
        </Plot>
        <Plot x={x2} y={y2} title="A*: priority g + Manhattan h" bare>
          <Raster x={XS} y={YS} z={za} scale="categorical" categoryNames={CATEGORIES} valueLabel="A*" />
        </Plot>
      </Plots>
    </Figure>
  )
}

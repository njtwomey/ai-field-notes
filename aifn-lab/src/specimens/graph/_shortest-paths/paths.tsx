import { aStarSteps, dijkstraSteps, shortestPath, type DijkstraState } from 'aifn/graph/shortest-paths'
import { fromEdges, heapSorted, type Graph } from 'aifn/graph'
import { stream, uniform } from 'aifn/foundation/random'
import { trace, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Slider } from '@lab/controls'
import { Columns, Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout } from '@lab/viz'
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

function Panel({ walls, s, name }: { walls: boolean[]; s: DijkstraState; name: string }) {
  const z = useMemo(() => grid(walls, s), [walls, s])
  return (
    <ChartSize scale={0.8}>
      <Heatmap
        x={XS}
        y={YS}
        z={z}
        scale="categorical"
        categoryNames={CATEGORIES}
        valueLabel={name}
        equalAspect
        zoom={false}
      />
    </ChartSize>
  )
}

function queueItems(s: DijkstraState): string[] {
  return liveQueue(s)
    .slice(0, 5)
    .map((e) => `${cell(e.value)} ${e.priority}`)
}

export function MazeSpecimen() {
  const [seed, setSeed] = useState(3)
  const [density, setDensity] = useState(0.28)
  const { walls, graph } = useMemo(() => maze(seed, density), [seed, density])
  const traces = useMemo((): [Trace<DijkstraState>, Trace<DijkstraState>] => {
    const n = 4 * H * W
    return [
      trace(dijkstraSteps(graph, { source: SOURCE, target: TARGET }), undefined, n),
      trace(aStarSteps(graph, { source: SOURCE, target: TARGET, heuristic: manhattan }), undefined, n),
    ]
  }, [graph])
  const count = Math.max(traces[0].steps.length, traces[1].steps.length)
  const [step, setStep] = useState(count - 1)
  const at = Math.min(step, count - 1)
  const d = stateAt(traces[0], at)
  const a = stateAt(traces[1], at)
  const length = (s: DijkstraState) => (s.done ? s.distance.data[TARGET] : NaN)

  return (
    <Figure
      title="Dijkstra against A* on a grid maze"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <>
          <Slider label="maze" value={seed} onChange={setSeed} min={1} max={30} step={1} />
          <Slider label="wall density" value={density} onChange={setDensity} min={0.1} max={0.4} step={0.02} />
          <div className="col-span-full">
            <Player label="expansion" value={at} onChange={setStep} count={count} defaultSpeed={12} />
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="Dijkstra expanded" value={d.expanded} />
          <Readout label="A* expanded" value={a.expanded} />
          <Readout
            label="shortest path"
            value={
              !a.done
                ? '…'
                : Number.isFinite(length(a))
                  ? `${length(a)} steps (Dijkstra ${Number.isFinite(length(d)) ? length(d) : '…'})`
                  : 'no path'
            }
          />
        </>
      }
      caption="From the top-left cell to the bottom-right, moving in four directions at unit cost. Each step pops the queued cell of least priority and expands it: Dijkstra's priority is the distance so far, A*'s adds the Manhattan distance to the goal, which never overestimates. Both find a shortest path (shown once done); A* expands fewer cells. Below each grid, the front of its priority queue as (row,col) priority."
    >
      <Columns
        panels={[
          {
            title: 'Dijkstra: priority g',
            body: <Panel walls={walls} s={d} name="Dijkstra" />,
            footer: <Sequence label="queue" ends="least first" items={queueItems(d)} />,
          },
          {
            title: 'A*: priority g + Manhattan h',
            body: <Panel walls={walls} s={a} name="A*" />,
            footer: <Sequence label="queue" ends="least first" items={queueItems(a)} />,
          },
        ]}
      />
    </Figure>
  )
}

/**
 * Figure plumbing shared by the graph-learning-tasks notes: a fixed plot height inside a Figure, a training run sliced
 * between frames, a player position that resets with its setup, and the layers that draw a small graph.
 */
import { useContext, useEffect, useState, type ReactElement, type ReactNode } from 'react'
import { FrameContext, Points, Segments } from 'aifn-render'

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
export function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Milliseconds of work per slice, so the page stays responsive while a run fills in. */
const SLICE = 25
/** Checkpoints per slice at most, so a slice ends even where the clock does not advance during a task. */
const MAX_PER_SLICE = 50

/**
 * Runs a generator of checkpoints in short slices between frames and returns the checkpoints so far, keyed to the
 * setup that started it: a new setup restarts the run. A generator that throws ends the run with its error.
 */
export function useSlicedRun<S, C>(setup: S, run: (setup: S) => Generator<C>) {
  const [state, setState] = useState<{ setup: S; checkpoints: C[]; done: boolean; error: string | null }>({
    setup,
    checkpoints: [],
    done: false,
    error: null,
  })
  useEffect(() => {
    const it = run(setup)
    const checkpoints: C[] = []
    let timer = 0
    const slice = () => {
      const start = performance.now()
      let done = false
      let error: string | null = null
      try {
        for (let k = 0; k < MAX_PER_SLICE && performance.now() - start < SLICE; k++) {
          const next = it.next()
          if (next.done) {
            done = true
            break
          }
          checkpoints.push(next.value)
        }
      } catch (e) {
        console.error(e)
        error = e instanceof Error ? e.message : String(e)
        done = true
      }
      setState({ setup, checkpoints: [...checkpoints], done, error })
      if (!done) timer = window.setTimeout(slice, 0)
    }
    timer = window.setTimeout(slice, 0)
    return () => window.clearTimeout(timer)
  }, [setup, run])
  return state.setup === setup ? state : { setup, checkpoints: [] as C[], done: false, error: null }
}

/** A player's position, back at 0 whenever the setup changes. */
export function usePlayerPosition<S>(setup: S): [number, (at: number) => void] {
  const [pos, setPos] = useState({ setup, at: 0 })
  return [pos.setup === setup ? pos.at : 0, (at: number) => setPos({ setup, at })]
}

/** A drawn graph: node positions and undirected edges as index pairs. */
export type Drawing = { pos: readonly (readonly [number, number])[]; edges: readonly (readonly [number, number])[] }

/** Positions of n nodes evenly on a circle of the given radius about (cx, cy), node 0 at the top. */
export function circleLayout(n: number, cx = 0, cy = 0, radius = 1): [number, number][] {
  return Array.from({ length: n }, (_, i) => {
    const a = Math.PI / 2 - (2 * Math.PI * i) / n
    return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)] as [number, number]
  })
}

/** Options of {@link graphLayers}. */
export type GraphLayerOptions = {
  /** Series id prefix, unique per plot. */
  id: string
  /** A class per node (palette slot k for class k); absent, nodes are muted. */
  classes?: ArrayLike<number>
  classNames?: readonly string[]
  /** A colour per node (e.g. from `useScaleColor`), in place of classes. */
  colors?: readonly string[]
  /** Text beside each node. */
  labels?: readonly (string | null)[]
  /** Nodes ringed in ink. */
  marked?: readonly number[]
  /** Edges drawn dashed and muted (indices into `edges`). */
  faded?: ReadonlySet<number>
  /** Node marker size in pixels (default 14). */
  size?: number
  /** Edge width in pixels (default 1.5). */
  edgeWidth?: number
}

/** The layers that draw a graph in a Plot: edges, then any marked rings, then the nodes. */
export function graphLayers(d: Drawing, o: GraphLayerOptions): ReactElement[] {
  const seg = (keep: (k: number) => boolean) =>
    d.edges.flatMap(([a, b], k) => (keep(k) ? [{ from: d.pos[a], to: d.pos[b] }] : []))
  const solid = seg((k) => !o.faded?.has(k))
  const faded = seg((k) => !!o.faded?.has(k))
  const x = d.pos.map((p) => p[0])
  const y = d.pos.map((p) => p[1])
  const size = o.size ?? 14
  const layers: ReactElement[] = []
  if (solid.length > 0)
    layers.push(
      <Segments key={`${o.id}-e`} id={`${o.id}-e`} name="edges" segments={solid} width={o.edgeWidth ?? 1.5} />,
    )
  if (faded.length > 0)
    layers.push(<Segments key={`${o.id}-f`} id={`${o.id}-f`} name="removed edges" segments={faded} dashed muted />)
  if (o.marked && o.marked.length > 0)
    layers.push(
      <Points
        key={`${o.id}-m`}
        id={`${o.id}-m`}
        name="marked"
        x={o.marked.map((v) => x[v])}
        y={o.marked.map((v) => y[v])}
        emphasis
        size={size + 9}
      />,
    )
  layers.push(
    <Points
      key={`${o.id}-n`}
      id={`${o.id}-n`}
      name="nodes"
      x={x}
      y={y}
      group={o.colors ? undefined : o.classes}
      groupNames={o.classNames}
      colors={o.colors}
      muted={!o.classes && !o.colors}
      labels={o.labels ?? undefined}
      size={size}
    />,
  )
  return layers
}

/** The axis ranges that fit a drawing, padded on every side. */
export function drawingRange(ds: readonly Drawing[], pad = 0.4): { x: [number, number]; y: [number, number] } {
  const xs = ds.flatMap((d) => d.pos.map((p) => p[0]))
  const ys = ds.flatMap((d) => d.pos.map((p) => p[1]))
  return {
    x: [Math.min(...xs) - pad, Math.max(...xs) + pad],
    y: [Math.min(...ys) - pad, Math.max(...ys) + pad],
  }
}

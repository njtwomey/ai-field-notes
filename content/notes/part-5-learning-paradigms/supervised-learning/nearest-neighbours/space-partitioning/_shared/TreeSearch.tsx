/**
 * Space-partitioning trees in the plane: how a k-d, ball or vantage-point tree partitions space (`TreeBuild`, one split
 * per step) and how an exact query searches it by branch and bound (`TreeSearch`, one node visit per step). Each figure
 * pairs the space with the tree itself, drawn as dots coloured by what the step does to each node, and explains the
 * step's decision below the space: the spreads that chose a k-d split, the leaf-size rule, the path from the root and,
 * when searching, the bound against the k-th best distance. Hovering a node outlines its region. The trees, the query
 * and its visit trace are aifn's (`aifn/numerics/neighbours`); this file only draws them. Shared by the notes of the
 * space-partitioning category.
 */
import { useMemo, useState, type ReactNode } from 'react'
import { stream, type Stream } from 'aifn/foundation/random'
import { fromData, toRows, type Tensor } from 'aifn/foundation/tensor'
import { treeFromParents } from 'aifn/graph'
import { ballTree, kdTree, treeQuery, vpTree, type SpaceTree } from 'aifn/numerics/neighbours'
import { anisotropicBlobs, blobs, circles, moons, spirals, swissRoll2d, xor } from 'aifn-methods/data/synthetic'
import {
  choice,
  ControlRow,
  Curve,
  Figure,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  row,
  Segments,
  Shapes,
  TreeView,
  useAxis,
  useFigureState,
  type ElementState,
  type Tone,
  type Vec2,
} from 'aifn-render'

export type TreeKind = 'kd' | 'ball' | 'vp'

const f3 = (v: number) => (Number.isFinite(v) ? String(Number(v.toPrecision(3))) : '—')
const RANGE: [number, number] = [-12, 12]
const FRAME = {
  x: [RANGE[0], RANGE[1], RANGE[1], RANGE[0], RANGE[0]],
  y: [RANGE[0], RANGE[0], RANGE[1], RANGE[1], RANGE[0]],
}
const KINDS = [
  { value: 'kd' as const, label: 'k-d tree' },
  { value: 'ball' as const, label: 'ball tree' },
  { value: 'vp' as const, label: 'vantage-point tree' },
]
const BUILD = { kd: kdTree, ball: ballTree, vp: vpTree }
const SUB = ['₁', '₂']

type Generator = (s: Stream, n: number) => { x: unknown }
const DATASETS: { value: string; label: string; make: Generator }[] = [
  { value: 'blobs', label: 'clusters', make: (s, n) => blobs(s, { n, centers: 5, dim: 2, box: [-8, 8] }) },
  { value: 'stretched', label: 'stretched clusters', make: (s, n) => anisotropicBlobs(s, { n }) },
  { value: 'uniform', label: 'uniform square', make: (s, n) => xor(s, { n, kind: 'uniform' } as never) },
  { value: 'moons', label: 'two moons', make: (s, n) => moons(s, { n } as never) },
  { value: 'circles', label: 'nested circles', make: (s, n) => circles(s, { n } as never) },
  { value: 'spirals', label: 'spirals', make: (s, n) => spirals(s, { n } as never) },
  { value: 'roll', label: 'swiss roll', make: (s, n) => swissRoll2d(s, { n } as never) },
]

/** The chosen data set, centred and scaled into the square [−10, 10]² (aspect kept), so every set fills the plot. */
function usePoints(dataset: string, n: number) {
  return useMemo(() => {
    const gen = DATASETS.find((d) => d.value === dataset) ?? DATASETS[0]
    const raw = toRows(gen.make(stream(`notes/space-partitioning/${dataset}`), n).x as Tensor) as number[][]
    const mid = [0, 1].map((c) => (Math.min(...raw.map((r) => r[c])) + Math.max(...raw.map((r) => r[c]))) / 2)
    const half = Math.max(...raw.flatMap((r) => [Math.abs(r[0] - mid[0]), Math.abs(r[1] - mid[1])])) || 1
    const rows = raw.map((r) => [((r[0] - mid[0]) / half) * 10, ((r[1] - mid[1]) / half) * 10])
    return {
      x: fromData(Float64Array.from(rows.flat()), [rows.length, 2]),
      xs: rows.map((r) => r[0]),
      ys: rows.map((r) => r[1]),
    }
  }, [dataset, n])
}

type Poly = { x: number[]; y: number[] }
type Cell = { lo: number[]; hi: number[] }

const circle = (cx: number, cy: number, r: number, steps = 64): Poly => {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= steps; i++) {
    x.push(cx + r * Math.cos((2 * Math.PI * i) / steps))
    y.push(cy + r * Math.sin((2 * Math.PI * i) / steps))
  }
  return { x, y }
}

const rect = (lo: readonly number[], hi: readonly number[]): Poly => ({
  x: [lo[0], hi[0], hi[0], lo[0], lo[0]],
  y: [lo[1], lo[1], hi[1], hi[1], lo[1]],
})

/** NaN-separated polyline of several outlines, so one layer draws them all. */
const joined = (parts: Poly[]): Poly => ({
  x: parts.flatMap((p) => [...p.x, NaN]),
  y: parts.flatMap((p) => [...p.y, NaN]),
})

/** A filled region (the first closed outline of a polyline). */
const fill = (p: Poly, tone: number | 'ink', opacity: number) => {
  const end = p.x.findIndex((v) => Number.isNaN(v))
  const k = end < 0 ? p.x.length : end
  return { contours: [p.x.slice(0, k).map((x, i) => [x, p.y[i]] as const)], tone, opacity }
}

/**
 * The region each k-d node owns: the plot's square cut by the splits on the path from the root (the cells tile the
 * plane, unlike the tight box of a node's points, which the search's bound uses).
 */
function kdCells(tree: SpaceTree): Cell[] {
  const cells: Cell[] = []
  const walk = (id: number, lo: number[], hi: number[]) => {
    cells[id] = { lo, hi }
    const nd = tree.nodes[id]
    if (nd.left < 0) return
    const j = nd.dim
    walk(
      nd.left,
      lo,
      hi.map((v, c) => (c === j ? nd.split : v)),
    )
    walk(
      nd.right,
      lo.map((v, c) => (c === j ? nd.split : v)),
      hi,
    )
  }
  if (tree.nodes.length) walk(0, [RANGE[0], RANGE[0]], [RANGE[1], RANGE[1]])
  return cells
}

/** Internal nodes in breadth-first order: the order in which the build figure applies the splits. */
function splitOrder(tree: SpaceTree): number[] {
  const out: number[] = []
  const queue = tree.nodes.length ? [0] : []
  while (queue.length) {
    const id = queue.shift()!
    const nd = tree.nodes[id]
    if (nd.left < 0) continue
    out.push(id)
    queue.push(nd.left, nd.right)
  }
  return out
}

/** The point indices of a node. */
const pointsOf = (tree: SpaceTree, id: number) =>
  Array.from(tree.order.subarray(tree.nodes[id].start, tree.nodes[id].end))

/** The range of each coordinate over a node's points. */
const spreadOf = (tree: SpaceTree, id: number) =>
  [0, 1].map((c) => {
    const v = pointsOf(tree, id).map((i) => tree.data[i * 2 + c])
    return { lo: Math.min(...v), hi: Math.max(...v) }
  })

/** The outline the search's bound is computed from: the tight box (k-d), the ball, or the shell (vantage point). */
function boundOutline(tree: SpaceTree, id: number): Poly {
  const nd = tree.nodes[id]
  if (tree.kind === 'kd-tree') return rect(nd.lower!, nd.upper!)
  if (tree.kind === 'ball-tree') return circle(nd.centre![0], nd.centre![1], nd.radius!)
  if (!nd.anchor || !nd.shell) return FRAME
  const [cx, cy] = nd.anchor
  return joined([circle(cx, cy, nd.shell[0]), circle(cx, cy, nd.shell[1])])
}

/** A node's region for hovering: its k-d cell, else the outline its bound uses. */
const regionOf = (tree: SpaceTree, cells: Cell[], id: number) =>
  tree.kind === 'kd-tree' ? rect(cells[id].lo, cells[id].hi) : boundOutline(tree, id)

/** One line about a node, for the hover readout. */
function describe(tree: SpaceTree, id: number): string {
  const nd = tree.nodes[id]
  const n = nd.end - nd.start
  if (nd.left < 0) return `leaf, ${n} points`
  if (tree.kind === 'kd-tree') return `split x${SUB[nd.dim]} at ${f3(nd.split)}, ${n} points`
  if (tree.kind === 'ball-tree') return `ball of radius ${f3(nd.radius!)}, ${n} points`
  return `vantage point ${nd.vantage}, μ = ${f3(nd.split)}, ${n} points`
}

/** The test on the edge from a node to its child, as the path from the root reads it. */
function edgeTest(tree: SpaceTree, parent: number, child: number): string {
  const nd = tree.nodes[parent]
  const left = child === nd.left
  if (tree.kind === 'kd-tree') return `x${SUB[nd.dim]} ${left ? '<' : '≥'} ${f3(nd.split)}`
  if (tree.kind === 'ball-tree') return left ? 'first half' : 'second half'
  return `${left ? 'inside' : 'outside'} μ = ${f3(nd.split)} of point ${nd.vantage}`
}

/** The tree as an `aifn/graph` tree (children in creation order: left, then right) and each node's parent. */
function useGraph(tree: SpaceTree) {
  return useMemo(() => {
    const parents: (number | null)[] = tree.nodes.map(() => null)
    tree.nodes.forEach((nd, id) => {
      if (nd.left >= 0) {
        parents[nd.left] = id
        parents[nd.right] = id
      }
    })
    return { graph: treeFromParents(parents), parents }
  }, [tree])
}

/** Root first. */
const pathTo = (parents: (number | null)[], id: number) => {
  const out: number[] = []
  for (let v: number | null = id; v !== null; v = parents[v]) out.unshift(v)
  return out
}

/** The path from the root to a node as its chain of tests. */
function PathLine({ tree, parents, id }: { tree: SpaceTree; parents: (number | null)[]; id: number }) {
  const path = pathTo(parents, id)
  const tests = path.slice(1).map((c, i) => edgeTest(tree, path[i], c))
  return (
    <div>
      <span className="text-muted-foreground">path: </span>
      {tests.length ? `root → ${tests.join(' → ')}` : 'the root'}
      <span className="text-muted-foreground"> (depth {path.length - 1})</span>
    </div>
  )
}

const schema = (index: TreeKind) => ({
  tree: row('1 · data and tree', {
    dataset: choice(
      DATASETS.map(({ value, label }) => ({ value, label })),
      'blobs',
      { label: 'data' },
    ),
    n: int(240, { ge: 20, le: 3000, suggestions: [120, 240, 1000], label: 'points' }),
    kind: choice(KINDS, index, { label: 'index' }),
    leafSize: int(6, { ge: 1, le: 100, suggestions: [2, 6, 16, 40], label: 'leaf size' }),
  }),
})

function useTree(dataset: string, kind: TreeKind, leafSize: number, n: number) {
  const pts = usePoints(dataset, n)
  const tree = useMemo(() => BUILD[kind](pts.x, { leafSize }), [pts, kind, leafSize])
  const cells = useMemo(() => (tree.kind === 'kd-tree' ? kdCells(tree) : []), [tree])
  return { pts, tree, cells }
}

/** The hovered node's region over the plot: a light fill and a bold outline. */
function HoverRegion({ region }: { region: Poly | null }) {
  if (!region) return null
  return (
    <>
      <Shapes name="hovered node" shapes={[fill(region, 'ink', 0.06)]} />
      <Curve name="hovered node" x={region.x} y={region.y} emphasis width={2.5} silent />
    </>
  )
}

/**
 * The space and the tree: side by side when the figure is wide enough (a container query, so it follows the figure's
 * width, not the window's), stacked in a narrow note column.
 */
function SpaceAndTree({ space, footer, tree }: { space: ReactNode; footer: ReactNode; tree: ReactNode }) {
  return (
    <div className="@container">
      <div className="grid grid-cols-1 gap-4 @3xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-2">
          <div className="text-xs font-medium text-muted-foreground">space</div>
          <div className="aspect-square w-full">{space}</div>
          <div className="text-xs">{footer}</div>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <div className="text-xs font-medium text-muted-foreground">tree</div>
          <div className="h-72 @3xl:h-auto @3xl:min-h-0 @3xl:flex-1">{tree}</div>
        </div>
      </div>
    </div>
  )
}

const Footer = ({ children }: { children: ReactNode }) => (
  <div className="flex flex-col gap-0.5 leading-snug">{children}</div>
)

/** Why the build made this split: the spreads (k-d and ball), the vantage point (vp), and the leaf-size rule. */
function BuildDecision({
  tree,
  parents,
  id,
  leafSize,
}: {
  tree: SpaceTree
  parents: (number | null)[]
  id: number
  leafSize: number
}) {
  if (id < 0)
    return (
      <Footer>
        <div>
          The root holds all {tree.n} points. A node is split while it holds more than the leaf size, {leafSize} points;
          play to apply the splits.
        </div>
      </Footer>
    )
  const nd = tree.nodes[id]
  const n = nd.end - nd.start
  const nl = tree.nodes[nd.left].end - tree.nodes[nd.left].start
  const nr = n - nl
  const leafNote = (m: number) => (m <= leafSize ? 'a leaf' : 'split again later')
  let why: ReactNode
  if (tree.kind === 'vp-tree') {
    why = (
      <div>
        Vantage point: point {nd.vantage}, the node&apos;s point farthest from its centroid. The node&apos;s points are
        sorted by distance from it and split at the median, μ = {f3(nd.split)}.
      </div>
    )
  } else {
    const s = spreadOf(tree, id)
    const w = s.map((r) => r.hi - r.lo)
    const j = w[1] > w[0] ? 1 : 0
    why = (
      <div>
        Spread of the node&apos;s points: x₁ over [{f3(s[0].lo)}, {f3(s[0].hi)}] (width {f3(w[0])}), x₂ over [
        {f3(s[1].lo)}, {f3(s[1].hi)}] (width {f3(w[1])}). The wider coordinate, x{SUB[j]}, is split at its median,{' '}
        {f3(nd.split)}
        {tree.kind === 'ball-tree'
          ? `, and each half gets the smallest ball around its centroid that covers it (radii ${f3(tree.nodes[nd.left].radius!)} and ${f3(tree.nodes[nd.right].radius!)})`
          : ''}
        .
      </div>
    )
  }
  return (
    <Footer>
      <PathLine tree={tree} parents={parents} id={id} />
      {why}
      <div>
        Children: {nl} points ({leafNote(nl)}) and {nr} points ({leafNote(nr)}); a node with at most {leafSize} points
        is a leaf.
      </div>
    </Footer>
  )
}

// ── Building: how the space is partitioned ───────────────────────────────────────────────────────────────────────────

/** The construction, one split per step: the region being split, the split, and which child each point goes to. */
export function TreeBuild({ index = 'kd' }: { index?: TreeKind }) {
  const state = useFigureState(schema(index))
  const { dataset, kind, leafSize, n } = state.tree
  const { pts, tree, cells } = useTree(dataset, kind, leafSize, n)
  const { graph, parents } = useGraph(tree)
  const order = useMemo(() => splitOrder(tree), [tree])
  const [step, setStep] = useState(0)
  const [hover, setHover] = useState<number | null>(null)
  const at = Math.min(step, order.length)
  const done = useMemo(() => order.slice(0, at), [order, at])
  const currentId = at > 0 ? order[at - 1] : -1
  const current = currentId >= 0 ? tree.nodes[currentId] : null

  const view = useMemo(() => {
    const applied = new Set(done)
    let lines: { from: readonly [number, number]; to: readonly [number, number] }[] = []
    let outlines: Poly[] = []
    let currentSplit: Poly | null = null
    let region: Poly | null = null
    if (tree.kind === 'kd-tree') {
      lines = done.map((id) => {
        const nd = tree.nodes[id]
        const { lo, hi } = cells[id]
        return nd.dim === 0
          ? { from: [nd.split, lo[1]] as const, to: [nd.split, hi[1]] as const }
          : { from: [lo[0], nd.split] as const, to: [hi[0], nd.split] as const }
      })
    } else if (tree.kind === 'ball-tree') {
      // The balls of the current leaves: children of applied splits that have not been split themselves.
      outlines = done
        .flatMap((id) => [tree.nodes[id].left, tree.nodes[id].right])
        .filter((c) => !applied.has(c))
        .map((c) => boundOutline(tree, c))
    } else {
      outlines = done.map((id) => {
        const nd = tree.nodes[id]
        return circle(tree.data[nd.vantage! * 2], tree.data[nd.vantage! * 2 + 1], nd.split)
      })
    }
    if (current) {
      region = tree.kind === 'vp-tree' ? null : regionOf(tree, cells, currentId)
      if (tree.kind === 'kd-tree') {
        const { lo, hi } = cells[currentId]
        currentSplit =
          current.dim === 0
            ? { x: [current.split, current.split], y: [lo[1], hi[1]] }
            : { x: [lo[0], hi[0]], y: [current.split, current.split] }
      } else if (tree.kind === 'ball-tree') {
        currentSplit = joined([boundOutline(tree, current.left), boundOutline(tree, current.right)])
      } else {
        const v = current.vantage!
        currentSplit = circle(tree.data[v * 2], tree.data[v * 2 + 1], current.split)
      }
    }
    // The tree: nodes appear when their parent is split; the split node is active, its children take the points'
    // colours, nodes already split are ink and nodes still to split neutral.
    const visible = tree.nodes.map((_, id) => id === 0 || applied.has(parents[id]!))
    const tone: Tone[] = tree.nodes.map((_, id) =>
      current && id === current.left ? 0 : current && id === current.right ? 1 : applied.has(id) ? 'ink' : 'neutral',
    )
    const nodeState: ElementState[] = tree.nodes.map((_, id) =>
      id === currentId ? 'active' : visible[id] ? 'done' : 'idle',
    )
    return {
      lines,
      outlines: joined(outlines),
      currentSplit,
      region,
      visible,
      tone,
      nodeState,
      leftPts: current ? pointsOf(tree, current.left) : [],
      rightPts: current ? pointsOf(tree, current.right) : [],
    }
  }, [tree, done, cells, current, currentId, parents])

  const ax = useAxis({ label: 'x₁', range: RANGE })
  const ay = useAxis({ label: 'x₂', range: RANGE, equal: ax })
  const hovered = hover !== null && view.visible[hover] ? hover : null

  return (
    <Figure
      title="Building the tree: how the space is partitioned"
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · splits">
          <Player
            value={at}
            onChange={setStep}
            count={order.length + 1}
            label="split"
            format={(p) => (p === 0 ? 'the root' : `${p} of ${order.length}`)}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="regions" value={`${1 + at} of ${order.length + 1}`} />
          <Readout label="depth" value={current ? current.depth : 0} />
          <Readout label="hovered node" value={hovered !== null ? `${hovered}: ${describe(tree, hovered)}` : '—'} />
        </>
      }
      caption={
        <>
          The space and its tree, one dot per node. Each step splits one node, breadth first: the node is ringed in the
          tree with its path from the root, its region is shaded in the space, and its two children (and the points sent
          to each) take the two colours. The finished tree is drawn faintly from the start, and nodes darken as their
          parent is split. Below the space: why this split was made.{' '}
          {kind === 'kd'
            ? 'The k-d regions tile the plane: every point of space lies in exactly one leaf.'
            : kind === 'ball'
              ? 'Ball-tree balls may overlap and cover only the data, not all of space.'
              : 'A vantage-point child is its parent’s region inside or outside the circle.'}{' '}
          Hover a node of the tree to outline its region.
        </>
      }
    >
      <SpaceAndTree
        space={
          <Plot x={ax} y={ay} legend={false}>
            {view.region && <Shapes name="region being split" shapes={[fill(view.region, 0, 0.1)]} />}
            <Curve x={FRAME.x} y={FRAME.y} muted width={1} silent />
            {view.lines.length > 0 && <Segments name="splits so far" segments={view.lines} />}
            <Curve
              name={kind === 'ball' ? 'leaf balls' : 'splits so far'}
              x={view.outlines.x}
              y={view.outlines.y}
              muted
              width={1}
              silent
            />
            <Points name="points" x={pts.xs} y={pts.ys} muted thin />
            <Points
              name="to the first child"
              x={view.leftPts.map((i) => pts.xs[i])}
              y={view.leftPts.map((i) => pts.ys[i])}
              slot={0}
              size={5}
            />
            <Points
              name="to the second child"
              x={view.rightPts.map((i) => pts.xs[i])}
              y={view.rightPts.map((i) => pts.ys[i])}
              slot={1}
              size={5}
            />
            {view.currentSplit && (
              <Curve name="this split" x={view.currentSplit.x} y={view.currentSplit.y} emphasis width={2} silent />
            )}
            {kind === 'vp' && current && (
              <Points
                name="vantage point"
                x={[tree.data[current.vantage! * 2]]}
                y={[tree.data[current.vantage! * 2 + 1]]}
                emphasis
                size={9}
              />
            )}
            <HoverRegion region={hovered !== null ? regionOf(tree, cells, hovered) : null} />
          </Plot>
        }
        footer={<BuildDecision tree={tree} parents={parents} id={currentId} leafSize={leafSize} />}
        tree={
          <TreeView
            tree={graph}
            ariaLabel={`${KINDS.find((k) => k.value === kind)!.label} under construction`}
            height="fill"
            shape="dot"
            nodeLabels={() => ''}
            nodeSummary={false}
            nodeTone={view.tone}
            nodeState={view.nodeState}
            highlight={currentId >= 0 ? pathTo(parents, currentId) : []}
            onNodeHover={setHover}
          />
        }
      />
    </Figure>
  )
}

// ── Searching: one query by branch and bound ─────────────────────────────────────────────────────────────────────────

/** An exact k-NN query by branch and bound, one node visit per step; `index` picks the tree it opens with. */
export function TreeSearch({ index = 'kd' }: { index?: TreeKind }) {
  const state = useFigureState({
    ...schema(index),
    search: row('2 · search', { k: int(5, { ge: 1, le: 50, suggestions: [1, 5, 10], label: 'k' }) }),
  })
  const { dataset, kind, leafSize, n } = state.tree
  const { k } = state.search
  const { pts, tree, cells } = useTree(dataset, kind, leafSize, n)
  const { graph, parents } = useGraph(tree)
  const [query, setQuery] = useState<Vec2>([1.5, -0.5])
  const [hover, setHover] = useState<number | null>(null)
  const kk = Math.min(k, n)
  const result = useMemo(() => treeQuery(tree, query, kk), [tree, query, kk])
  const events = result.visits
  const [step, setStep] = useState(0)
  const at = Math.min(step, events.length)
  const seen = useMemo(() => events.slice(0, at), [events, at])
  const current = at > 0 ? events[at - 1] : null

  // The whole partition, faint: k-d splits across their cells, ball-tree leaf balls, vantage-point circles.
  const partition = useMemo(() => {
    if (tree.kind === 'kd-tree')
      return {
        lines: tree.nodes.flatMap((nd, id) => {
          if (nd.left < 0) return []
          const { lo, hi } = cells[id]
          return [
            nd.dim === 0
              ? { from: [nd.split, lo[1]] as const, to: [nd.split, hi[1]] as const }
              : { from: [lo[0], nd.split] as const, to: [hi[0], nd.split] as const },
          ]
        }),
        curves: joined([]),
      }
    const parts = tree.nodes.flatMap((nd, id) => {
      if (tree.kind === 'ball-tree') return nd.left < 0 ? [boundOutline(tree, id)] : []
      return nd.left < 0 ? [] : [circle(tree.data[nd.vantage! * 2], tree.data[nd.vantage! * 2 + 1], nd.split)]
    })
    return { lines: [], curves: joined(parts) }
  }, [tree, cells])

  const drawn = useMemo(() => {
    const scanIds = seen.filter((v) => v.action === 'scan').map((v) => v.node)
    const pruneIds = seen.filter((v) => v.action === 'prune').map((v) => v.node)
    const region = (id: number) => regionOf(tree, cells, id)
    // The tree: unvisited nodes dimmed; descended ink, scanned leaves colour 1, pruned nodes colour 2 (dashed); the
    // node of this step active, and the path from the root to it highlighted.
    const action = new Map<number, string>()
    for (const v of seen) action.set(v.node, v.action)
    const tone: Tone[] = tree.nodes.map((_, id) => {
      const a = action.get(id)
      return a === 'scan' ? 0 : a === 'prune' ? 1 : a === 'descend' ? 'ink' : 'neutral'
    })
    const nodeState: ElementState[] = tree.nodes.map((_, id) =>
      current && id === current.node ? 'active' : action.has(id) ? 'done' : 'idle',
    )
    return {
      scannedFill: tree.kind === 'kd-tree' ? scanIds.map((id) => fill(region(id), 0, 0.12)) : [],
      scanned: joined(scanIds.map(region)),
      pruned: joined(pruneIds.map(region)),
      points: scanIds.flatMap((id) => pointsOf(tree, id)),
      pruneCount: pruneIds.length,
      prunedSet: new Set(pruneIds),
      tone,
      nodeState,
    }
  }, [seen, tree, cells, current])
  const worst = current ? current.worst : Infinity
  const before = at > 1 ? events[at - 2].worst : Infinity
  const done = at === events.length
  const answer = done ? result.indices : []
  const ax = useAxis({ label: 'x₁', range: RANGE })
  const ay = useAxis({ label: 'x₂', range: RANGE, equal: ax })
  const radius = Number.isFinite(worst) ? circle(query[0], query[1], worst) : null
  const cur = current ? boundOutline(tree, current.node) : null
  const boundName = kind === 'kd' ? 'box of its points' : kind === 'ball' ? 'ball' : 'shell'
  const hoveredVisit = hover !== null ? seen.filter((v) => v.node === hover).at(-1) : undefined

  const decision = !current ? (
    <Footer>
      <div>
        The search starts at the root with no candidates, so the k-th best distance r is ∞. Each node is first bounded:
        the distance from the query to the node&apos;s {boundName} is no more than the distance to any of its points.
      </div>
    </Footer>
  ) : (
    <Footer>
      <PathLine tree={tree} parents={parents} id={current.node} />
      <div>
        Node {current.node}: bound {f3(current.bound)} against r = {f3(before)} before this visit.{' '}
        {current.action === 'prune'
          ? `The bound exceeds r, so no point of the node can enter the ${kk} best: the node and its whole subtree are skipped.`
          : current.action === 'scan'
            ? `The bound is below r and the node is a leaf (at most ${leafSize} points): its ${tree.nodes[current.node].end - tree.nodes[current.node].start} distances are computed, and r becomes ${f3(current.worst)}.`
            : 'The bound is below r, so the search descends: both children are bounded and the nearer one is visited first.'}
      </div>
      {done && <div>Search complete: the {kk} answers are joined to the query.</div>}
    </Footer>
  )

  return (
    <Figure
      title="Searching the tree: exact k-NN by branch and bound"
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · visits">
          <Player
            value={at}
            onChange={setStep}
            count={events.length + 1}
            label="visit"
            format={(p) => (p === 0 ? 'start' : `${p}: ${events[p - 1]?.action ?? ''}`)}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="k-th best distance" value={f3(worst)} />
          <Readout
            label="distances in total"
            value={`${result.distanceEvaluations} of ${n} (${f3((100 * result.distanceEvaluations) / n)}%)`}
          />
          <Readout label="nodes pruned" value={drawn.pruneCount} />
          <Readout
            label="hovered node"
            value={
              hover !== null
                ? `${hover}: ${describe(tree, hover)}${hoveredVisit ? `; ${hoveredVisit.action}, bound ${f3(hoveredVisit.bound)}` : '; not visited yet'}`
                : '—'
            }
          />
        </>
      }
      caption={
        <>
          The space and its tree. Each step visits one node, nearer child first, along the ringed node&apos;s path from
          the root. A scanned leaf (colour 1{kind === 'kd' ? ', its region shaded' : ''}) adds its points to the
          candidates, and the dashed circle around the query is the k-th best distance so far. A node is pruned (colour
          2, dashed) when its lower bound, the distance from the query to the node&apos;s {boundName} (bold), already
          exceeds that circle; its whole subtree is skipped and stays faint in the tree. Below the space: the reason for
          this step. Drag the query; hover a node of the tree to outline its region.
        </>
      }
    >
      <SpaceAndTree
        space={
          <Plot x={ax} y={ay} legend={false}>
            {drawn.scannedFill.length > 0 && <Shapes name="scanned regions" shapes={drawn.scannedFill} />}
            <Curve x={FRAME.x} y={FRAME.y} muted width={1} silent />
            {partition.lines.length > 0 && <Segments name="splits" segments={partition.lines} />}
            <Curve
              name={kind === 'ball' ? 'leaf balls' : 'splits'}
              x={partition.curves.x}
              y={partition.curves.y}
              muted
              width={1}
              silent
            />
            <Points name="points" x={pts.xs} y={pts.ys} muted thin />
            <Curve name="scanned leaves" x={drawn.scanned.x} y={drawn.scanned.y} slot={0} width={1.5} silent />
            <Curve name="pruned nodes" x={drawn.pruned.x} y={drawn.pruned.y} slot={1} dashed width={1} silent />
            <Points
              name="distance computed"
              x={drawn.points.map((i) => pts.xs[i])}
              y={drawn.points.map((i) => pts.ys[i])}
              slot={0}
              size={5}
            />
            {cur && <Curve name="current node's bound" x={cur.x} y={cur.y} emphasis width={2} silent />}
            {radius && <Curve name="k-th best distance" x={radius.x} y={radius.y} emphasis dashed width={1} silent />}
            <Segments
              name="answer"
              segments={answer.map((i) => ({ from: query, to: [pts.xs[i], pts.ys[i]] as const }))}
              emphasis
            />
            <HoverRegion region={hover !== null ? regionOf(tree, cells, hover) : null} />
            <Handle kind="point" at={query} onDrag={setQuery} label="query" />
          </Plot>
        }
        footer={decision}
        tree={
          <TreeView
            tree={graph}
            ariaLabel={`${KINDS.find((x) => x.value === kind)!.label} being searched`}
            height="fill"
            shape="dot"
            nodeLabels={() => ''}
            nodeSummary={false}
            nodeTone={drawn.tone}
            nodeState={drawn.nodeState}
            node={(id) => (drawn.prunedSet.has(id) ? { dashed: true } : {})}
            highlight={current ? pathTo(parents, current.node) : []}
            onNodeHover={setHover}
          />
        }
      />
    </Figure>
  )
}

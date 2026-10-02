/**
 * Nearest-neighbour search in two dimensions with a draggable query: the k-d and ball trees' branch and bound, the
 * buckets of locality-sensitive hashing, HNSW's layered greedy descent, and a recall-against-speed benchmark in the
 * worker (`aifn-applied/retrieval/ann`). Every index, query and measurement is `aifn/numerics/neighbours`'.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  ballTree,
  bruteForceQuery,
  hnswIndex,
  hnswQuery,
  hyperplaneFamily,
  kdTree,
  lshIndex,
  lshQuery,
  pStableFamily,
  treeQuery,
  type SpaceTree,
} from 'aifn/numerics/neighbours'
import { blobs } from 'aifn-applied/data/synthetic'
import { ANN_METHODS, type AnnBenchmarkSnapshot, type AnnMethod } from 'aifn-applied/retrieval/ann'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, useStreamed } from '@lab/state'
import { formatValue } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Readout, Segments, useAxis, type Vec2 } from '@lab/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const RANGE: [number, number] = [-12, 12]

/** Clustered points in the plane: the data of every 2-D figure. */
function usePoints(n: number, seed: number) {
  return useMemo(() => {
    const data = blobs(stream(`lab/neighbours/${seed}`), { n, centers: 5, dim: 2, box: [-8, 8] })
    const x = data.x as Tensor
    const rows = toRows(x) as number[][]
    return { x, rows, xs: rows.map((r) => r[0]), ys: rows.map((r) => r[1]) }
  }, [n, seed])
}

const circle = (cx: number, cy: number, r: number, steps = 48) => {
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i <= steps; i++) {
    xs.push(cx + r * Math.cos((2 * Math.PI * i) / steps))
    ys.push(cy + r * Math.sin((2 * Math.PI * i) / steps))
  }
  return { x: xs, y: ys }
}

const box = (lo: readonly number[], hi: readonly number[]) => ({
  x: [lo[0], hi[0], hi[0], lo[0], lo[0]],
  y: [lo[1], lo[1], hi[1], hi[1], lo[1]],
})

/** A node's outline: its box (k-d) or its ball. */
const outline = (tree: SpaceTree, id: number) => {
  const nd = tree.nodes[id]
  return tree.kind === 'kd-tree' ? box(nd.lower!, nd.upper!) : circle(nd.centre![0], nd.centre![1], nd.radius!)
}

/** NaN-separated polyline of several outlines, so one layer draws them all. */
const joined = (parts: { x: number[]; y: number[] }[]) => ({
  x: parts.flatMap((p) => [...p.x, NaN]),
  y: parts.flatMap((p) => [...p.y, NaN]),
})

// ── Trees ────────────────────────────────────────────────────────────────────────────────────────────────────────────

export function TreeSearchFigure() {
  const state = useFigureState({
    tree: row('1 · tree', {
      kind: choice(
        [
          { value: 'kd', label: 'k-d tree' },
          { value: 'ball', label: 'ball tree' },
        ],
        'kd',
        { label: 'index' },
      ),
      leafSize: int(6, { ge: 1, le: 100, suggestions: [2, 6, 16, 40], label: 'leaf size' }),
    }),
    search: row('2 · search', {
      k: int(5, { ge: 1, le: 50, suggestions: [1, 5, 10], label: 'k' }),
      n: int(240, { ge: 20, le: 3000, suggestions: [120, 240, 1000], label: 'points' }),
    }),
  })
  const { kind, leafSize } = state.tree
  const { k, n } = state.search
  const pts = usePoints(n, 1)
  const [query, setQuery] = useState<Vec2>([1.5, -0.5])
  const tree = useMemo(
    () => (kind === 'kd' ? kdTree(pts.x, { leafSize }) : ballTree(pts.x, { leafSize })),
    [pts, kind, leafSize],
  )
  const kk = Math.min(k, n)
  const result = useMemo(() => treeQuery(tree, query, kk), [tree, query, kk])
  const events = result.visits
  const [step, setStep] = useState(0)
  const at = Math.min(step, events.length)
  const seen = events.slice(0, at)
  const current = at > 0 ? events[at - 1] : null

  // The whole partition, faint: k-d split lines inside their node's box, or every leaf's ball.
  const partition = useMemo(() => {
    if (tree.kind === 'kd-tree')
      return tree.nodes
        .filter((nd) => nd.dim >= 0)
        .map((nd) =>
          nd.dim === 0
            ? { from: [nd.split, nd.lower![1]] as const, to: [nd.split, nd.upper![1]] as const }
            : { from: [nd.lower![0], nd.split] as const, to: [nd.upper![0], nd.split] as const },
        )
    return []
  }, [tree])
  const leafBalls = useMemo(
    () =>
      tree.kind === 'ball-tree' ? joined(tree.nodes.flatMap((nd, i) => (nd.left < 0 ? [outline(tree, i)] : []))) : null,
    [tree],
  )
  const drawn = useMemo(() => {
    const scanned = seen.filter((v) => v.action === 'scan').map((v) => outline(tree, v.node))
    const pruned = seen.filter((v) => v.action === 'prune').map((v) => outline(tree, v.node))
    const scannedPoints = seen
      .filter((v) => v.action === 'scan')
      .flatMap((v) => Array.from(tree.order.subarray(tree.nodes[v.node].start, tree.nodes[v.node].end)))
    return { scanned: joined(scanned), pruned: joined(pruned), points: scannedPoints }
  }, [seen, tree])
  const worst = current ? current.worst : Infinity
  const done = at === events.length
  const answer = done ? result.indices : []
  const ax = useAxis({ label: 'x₀', range: RANGE })
  const ay = useAxis({ label: 'x₁', range: RANGE, equal: ax })
  const radius = Number.isFinite(worst) ? circle(query[0], query[1], worst) : null
  const cur = current ? outline(tree, current.node) : null

  return (
    <Figure
      title="Exact search by branch and bound: k-d trees and ball trees"
      purpose="A tree answers exactly while computing few distances: it visits the nearer child first and skips every node whose lower bound already exceeds the k-th best distance found."
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
          <Readout label="visit" value={current ? `${current.action} node ${current.node}` : 'none yet'} />
          <Readout label="node bound" value={current ? f3(current.bound) : '—'} />
          <Readout label="k-th best distance" value={f3(worst)} />
          <Readout label="distances so far" value={`${drawn.points.length} of ${n}`} />
          <Readout
            label="distances in total"
            value={`${result.distanceEvaluations} (${f3((100 * result.distanceEvaluations) / n)}%)`}
          />
          <Readout label="nodes pruned" value={seen.filter((v) => v.action === 'prune').length} />
          <Readout label="tree nodes" value={tree.nodes.length} />
        </>
      }
      caption={
        <>
          aifn <code>{kind === 'kd' ? 'kdTree' : 'ballTree'}</code> and <code>treeQuery</code> on {n} clustered points.
          Faint: the tree&apos;s partition (
          {kind === 'kd' ? 'median splits on the coordinate of largest spread' : 'every leaf’s ball'}). Play the search
          from its start: each scanned leaf (slot 1) adds its points to the candidates, the circle around the query is
          the k-th best distance so far, and a pruned node (dashed) is one whose {kind === 'kd' ? 'box' : 'ball'} lies
          wholly outside that circle. The k answers are joined to the query when the search ends. Drag the query; raise
          the leaf size to trade fewer nodes for more distances per leaf.
        </>
      }
    >
      <Plot x={ax} y={ay}>
        {partition.length > 0 && <Segments name="splits" segments={partition} />}
        {leafBalls && <Curve name="leaf balls" x={leafBalls.x} y={leafBalls.y} muted width={1} silent />}
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
        {cur && <Curve name="current node" x={cur.x} y={cur.y} emphasis width={2} silent />}
        {radius && <Curve name="k-th best distance" x={radius.x} y={radius.y} emphasis dashed width={1} silent />}
        <Segments
          name="answer"
          segments={answer.map((i) => ({ from: query, to: [pts.xs[i], pts.ys[i]] as const }))}
          emphasis
        />
        <Handle kind="point" at={query} onDrag={setQuery} label="query" />
      </Plot>
    </Figure>
  )
}

// ── LSH ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

export function LshFigure() {
  const state = useFigureState({
    family: row('1 · hash family', {
      family: choice(
        [
          { value: 'p-stable', label: 'p-stable (Euclidean)' },
          { value: 'hyperplane', label: 'random hyperplanes (cosine)' },
        ],
        'p-stable',
        { label: 'family' },
      ),
      hashes: int(2, { ge: 1, le: 12, suggestions: [1, 2, 4], label: 'hashes per table r' }),
      tables: int(4, { ge: 1, le: 32, suggestions: [1, 4, 8, 16], label: 'tables b' }),
      width: float(4, { gt: 0, le: 40, scale: 'log10', suggestions: [2, 4, 8], label: 'bucket width w' }),
    }),
    show: row('2 · show', {
      table: int(1, { ge: 1, le: 32, label: 'table drawn' }),
      k: int(5, { ge: 1, le: 30, label: 'k' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { family, hashes, tables, width } = state.family
  const { k, seed } = state.show
  const table = Math.min(state.show.table, tables) - 1
  const pts = usePoints(240, 1)
  const [query, setQuery] = useState<Vec2>([2, 1])
  const fam = useMemo(() => {
    const s = stream(`lab/lsh/${seed}`)
    return family === 'hyperplane'
      ? hyperplaneFamily(2, { tables, hashesPerTable: hashes, stream: s })
      : pStableFamily(2, { tables, hashesPerTable: hashes, width, stream: s })
  }, [family, tables, hashes, width, seed])
  const index = useMemo(() => lshIndex(pts.x, fam), [pts, fam])
  const r = useMemo(() => lshQuery(index, query, k), [index, query, k])
  const exact = useMemo(() => bruteForceQuery(pts.x, query, k, { metric: index.metric }), [pts, query, k, index])
  const inBucket = useMemo(() => index.tables[table][r.buckets[table]] ?? [], [index, table, r])
  const found = new Set(r.indices)
  const hits = exact.indices.filter((i) => found.has(i)).length
  // The hash boundaries of the drawn table: lines perpendicular to each projection.
  const lines = useMemo(() => {
    const P = toRows(fam.projections) as number[][]
    const off = fam.offsets ? toFlat(fam.offsets) : null
    const segs: { from: readonly [number, number]; to: readonly [number, number] }[] = []
    for (let j = table * hashes; j < (table + 1) * hashes; j++) {
      const [a, b] = P[j]
      const norm = Math.hypot(a, b)
      const ux = a / norm
      const uy = b / norm
      // Lines aᵀx = c: c = 0 for a hyperplane, c = w·t − b for each integer t for a p-stable hash.
      const levels =
        fam.family === 'hyperplane'
          ? [0]
          : Array.from({ length: 81 }, (_, t) => (t - 40) * fam.width! - off![j]).filter((c) => Math.abs(c / norm) < 20)
      for (const c of levels) {
        const px = (c / norm) * ux
        const py = (c / norm) * uy
        segs.push({ from: [px - 30 * uy, py + 30 * ux], to: [px + 30 * uy, py - 30 * ux] })
      }
    }
    return segs
  }, [fam, table, hashes])
  const ax = useAxis({ label: 'x₀', range: RANGE })
  const ay = useAxis({ label: 'x₁', range: RANGE, equal: ax })
  const sel = (ids: readonly number[]) => ({ x: ids.map((i) => pts.xs[i]), y: ids.map((i) => pts.ys[i]) })
  const cands = sel(r.candidates)
  const bucket = sel(inBucket)
  const ans = sel(r.indices)
  const missed = sel(exact.indices.filter((i) => !found.has(i)))
  return (
    <Figure
      title="Locality-sensitive hashing: buckets from random projections"
      purpose="Near points share a hash more often than far ones; ANDing r hashes into a bucket keeps far points out, and ORing b tables brings near points back in."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="query's bucket, drawn table" value={`${inBucket.length} points`} />
          <Readout label="candidates over all tables" value={`${r.candidates.length} of ${pts.xs.length}`} />
          <Readout label={`recall@${k}`} value={f3(hits / k)} />
          <Readout label="distances computed" value={r.distanceEvaluations} />
          <Readout label="bucket key" value={r.buckets[table]} />
        </>
      }
      caption={
        <>
          aifn <code>{family === 'hyperplane' ? 'hyperplaneFamily' : 'pStableFamily'}</code>, <code>lshIndex</code> and{' '}
          <code>lshQuery</code>: {tables} tables of {hashes} hashes. The lines are the drawn table&apos;s hash
          boundaries (
          {family === 'hyperplane'
            ? 'one line through the origin per hash: the sign of a random projection'
            : `the cells of ⌊(aᵀx + b)/w⌋ along each random direction, w = ${f3(width)}`}
          ); the points sharing the query&apos;s cell in that table are slot 1, the candidates from every table slot 2,
          and the {k} answers (reranked by exact {index.metric} distance) are ink marks. Exact neighbours the hashing
          missed are red. Drag the query; add tables to raise recall, add hashes per table to shrink the buckets.
        </>
      }
    >
      <Plot x={ax} y={ay}>
        <Segments name="hash boundaries" segments={lines} />
        <Points name="points" x={pts.xs} y={pts.ys} muted thin />
        <Points name="candidates (any table)" x={cands.x} y={cands.y} slot={1} size={5} />
        <Points name="query's bucket (drawn table)" x={bucket.x} y={bucket.y} slot={0} size={6} />
        <Points name={`answer (${k})`} x={ans.x} y={ans.y} emphasis size={9} />
        <Points name="missed exact neighbours" x={missed.x} y={missed.y} tone="destructive" shape={3} size={9} />
        <Handle kind="point" at={query} onDrag={setQuery} label="query" />
      </Plot>
    </Figure>
  )
}

// ── HNSW ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

export function HnswFigure() {
  const state = useFigureState({
    build: row('1 · build', {
      M: int(5, { ge: 2, le: 32, suggestions: [3, 5, 8, 16], label: 'links per point M' }),
      efConstruction: int(32, { ge: 1, le: 400, suggestions: [16, 32, 100], label: 'build beam' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    search: row('2 · search', {
      ef: int(8, { ge: 1, le: 200, suggestions: [1, 8, 32], label: 'beam width ef' }),
      k: int(5, { ge: 1, le: 30, label: 'k' }),
    }),
  })
  const { M, efConstruction, seed } = state.build
  const { ef, k } = state.search
  const pts = usePoints(300, 2)
  const [query, setQuery] = useState<Vec2>([-3, 4])
  const index = useMemo(
    () => hnswIndex(pts.x, { M, efConstruction, stream: stream(`lab/hnsw/${seed}`) }),
    [pts, M, efConstruction, seed],
  )
  const r = useMemo(() => hnswQuery(index, query, k, { ef }), [index, query, k, ef])
  // The search as one sequence of expansions, layer by layer from the top.
  const events = useMemo(() => r.layers.flatMap((l) => l.expanded.map((node) => ({ layer: l.layer, node }))), [r])
  const [step, setStep] = useState(0)
  const at = Math.min(step, events.length)
  const layers = Array.from({ length: Math.min(4, index.topLayer + 1) }, (_, i) => Math.min(index.topLayer, 3) - i)
  const shown = layers.map((l) => {
    const members: number[] = []
    for (let i = 0; i < index.n; i++) if (index.levels[i] >= l) members.push(i)
    const segs = members.flatMap((i) =>
      index.links[l][i]
        .filter((j) => j > i || !index.links[l][j].includes(i))
        .map((j) => ({
          from: [pts.xs[i], pts.ys[i]] as const,
          to: [pts.xs[j], pts.ys[j]] as const,
        })),
    )
    const path = events
      .slice(0, at)
      .filter((e) => e.layer === l)
      .map((e) => e.node)
    return { l, members, segs, path }
  })
  const done = at === events.length
  const exact = useMemo(() => bruteForceQuery(pts.x, query, k), [pts, query, k])
  const found = new Set(r.indices)
  const recall = exact.indices.filter((i) => found.has(i)).length / k
  const ax = useAxis({ label: 'x₀', range: RANGE })
  const ay = useAxis({ label: 'x₁', range: RANGE, equal: ax })
  const sel = (ids: readonly number[]) => ({ x: ids.map((i) => pts.xs[i]), y: ids.map((i) => pts.ys[i]) })
  const current = at > 0 ? events[at - 1] : null
  return (
    <Figure
      title="HNSW: a greedy walk down a hierarchy of proximity graphs"
      purpose="The sparse upper layers have long links that cross the space in a few hops; each layer's walk starts where the one above ended, and the dense bottom layer refines with a beam of width ef."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · search">
          <Player
            value={at}
            onChange={setStep}
            count={events.length + 1}
            label="expansion"
            format={(p) => (p === 0 ? 'entry' : `${p}: layer ${events[p - 1]?.layer}`)}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="layers" value={index.topLayer + 1} />
          <Readout
            label="expanding"
            value={current ? `node ${current.node} on layer ${current.layer}` : `entry ${index.entry}`}
          />
          <Readout label="distances (search)" value={`${r.distanceEvaluations} of ${index.n}`} />
          <Readout label="distances (build)" value={index.buildDistanceEvaluations} />
          <Readout label={`recall@${k}`} value={f3(recall)} />
        </>
      }
      caption={
        <>
          aifn <code>hnswIndex</code> (M = {M}, layer 0 allows {2 * M}) and <code>hnswQuery</code> on {index.n} points;
          panels show layers {layers.join(', ')} (top first; a point lives on every layer up to its random top layer).
          The walk (slot 1) expands one point at a time: on an upper layer it moves to the nearest neighbour until none
          is nearer, then drops a layer from there; on layer 0 the beam keeps the ef best. The answer (slot 0) is marked
          when the walk ends. Drag the query on any panel; with ef = 1 the bottom walk is greedy too and can stop short.
        </>
      }
    >
      <Plots cols={shown.length} scale={0.75}>
        {shown.map(({ l, members, segs, path }) => {
          const m = sel(members)
          const p = sel(path)
          return (
            <Plot key={l} x={ax} y={ay} title={`layer ${l} (${members.length} points)`}>
              <Segments name="links" segments={segs} />
              <Points name="points" x={m.x} y={m.y} muted thin />
              {p.x.length > 0 && <Curve name="walk" x={p.x} y={p.y} slot={1} showPoints />}
              <Points name="entry" x={[pts.xs[index.entry]]} y={[pts.ys[index.entry]]} emphasis size={8} />
              {l === 0 && done && <Points name="answer" x={sel(r.indices).x} y={sel(r.indices).y} slot={0} size={9} />}
              <Handle kind="point" at={query} onDrag={setQuery} label="query" />
            </Plot>
          )
        })}
      </Plots>
    </Figure>
  )
}

// ── Benchmark ────────────────────────────────────────────────────────────────────────────────────────────────────────

const METHOD_LABEL: Record<AnnMethod, string> = {
  'kd-tree': 'k-d tree (leaf size)',
  lsh: 'LSH (tables)',
  ivf: 'IVF (probes)',
  pq: 'PQ (codewords)',
  hnsw: 'HNSW (ef)',
}

export function BenchmarkFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(3000, { ge: 200, le: 20000, suggestions: [1000, 3000, 10000], label: 'points' }),
      dim: int(16, { ge: 2, le: 128, suggestions: [2, 8, 16, 64], label: 'dimensions' }),
      k: int(10, { ge: 1, le: 100, suggestions: [1, 10], label: 'k' }),
    }),
    show: row('2 · show', {
      cost: choice(
        [
          { value: 'qps', label: 'queries per second' },
          { value: 'distances', label: 'distances per query' },
        ],
        'qps',
        { label: 'cost axis' },
      ),
    }),
  })
  const { n, dim, k } = state.data
  const task = useMemo(
    () =>
      call<AnnBenchmarkSnapshot>('applied/retrieval/ann/annBenchmark', {
        data: call('applied/data/synthetic/blobs', call('foundation/random/stream', 'lab/ann-bench'), {
          n: n + 100,
          centers: 10,
          dim,
        }),
        queries: 100,
        k,
      }),
    [n, dim, k],
  )
  const run = useStreamed<AnnBenchmarkSnapshot>(task)
  const snap = run.value
  const qps = state.show.cost === 'qps'
  const ax = useAxis({ label: `recall@${k}`, range: [0, 1.02] })
  const ay = useAxis({
    label: qps ? 'queries per second (log)' : 'distances per query (log)',
    log: true,
    hold: 'union',
    key: `${n}-${dim}-${k}-${qps}`,
  })
  const value = (p: { queriesPerSecond: number; distancesPerQuery: number }) =>
    qps ? p.queriesPerSecond : p.distancesPerQuery
  return (
    <Figure
      title="Recall against speed, method by method"
      purpose="Every approximate index trades recall for speed through one knob; in a few dimensions trees win outright, and as dimensions grow the graph and quantisation indexes take over."
      state={state}
      defaultSize="L"
      readouts={
        snap
          ? [
              <Readout
                key="b"
                label="brute force"
                value={`${f3(snap.bruteForce.queriesPerSecond)} q/s, ${f3(snap.bruteForce.distancesPerQuery)} distances`}
              />,
              ...snap.curves.map((c) => (
                <Readout key={c.method} label={`${c.method} build`} value={`${f3(c.buildSeconds * 1000)} ms`} />
              )),
            ]
          : [<Readout key="w" label="status" value={run.running ? 'building…' : '—'} />]
      }
      caption={
        <>
          aifn-applied <code>annBenchmark</code> in the worker on {n} points from 10 Gaussian clusters in {dim}{' '}
          dimensions, 100 held-out queries. Each curve is one index of <code>aifn/numerics/neighbours</code> at several
          settings of its knob (labels in the tooltip), from cheap to thorough; brute force is the dashed line. Recall@k
          is against brute force. Times are this machine&apos;s; the distance counts are not, so switch the cost axis to
          compare without timing noise. Methods appear as they finish.
        </>
      }
    >
      <Plot x={ax} y={ay}>
        {snap && (
          <Curve
            name="brute force"
            x={[0, 1.02]}
            y={[value(snap.bruteForce), value(snap.bruteForce)]}
            emphasis
            dashed
            width={1}
          />
        )}
        {snap?.curves.map((c) => (
          <Curve
            key={c.method}
            name={METHOD_LABEL[c.method]}
            x={c.points.map((p) => p.recall)}
            y={c.points.map((p) => value(p))}
            slot={ANN_METHODS.indexOf(c.method)}
            showPoints
          />
        ))}
      </Plot>
    </Figure>
  )
}

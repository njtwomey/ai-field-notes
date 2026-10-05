/**
 * Showcase: decision trees, grown, inspected and pruned. Everything drawn is read from what aifn's fitter records: the
 * growth trace (`decisionTree().fit(…).training`, one node per step), each node's rows, class totals, split search
 * (each feature's best) and stop reason, the cost-complexity path with the node cut at each step, and the pure
 * readers `nodeRegion`, `decisionPath`, `keptNodes`, `decideTree` and `splitCurve` (the fitter's own split search, re-run
 * for one feature's full curve). The lab computes no tree maths.
 */
import { blobs, circles, moons, withLabelNoise, xor } from 'aifn-methods/data/synthetic'
import type { Dataset } from 'aifn-methods/data'
import {
  costComplexityPath,
  decideTree,
  decisionPath,
  decisionTree,
  keptNodes,
  nodePrediction,
  nodeRegion,
  splitCurve,
  type DecisionPath,
  type DecisionTree,
  type GrowthOrder,
  type LeafReason,
} from 'aifn-methods/learning/trees-and-ensembles'
import { stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset as toDataset } from 'aifn-compute/learning/estimators'
import { errorRate } from 'aifn-compute/learning/metrics'
import { useMemo, useState, type ReactNode } from 'react'
import { Player } from 'aifn-render/controls'
import { MathText } from 'aifn-render/diagram'
import { Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useFigureState, usePinned, type FigureState } from 'aifn-render/state'
import {
  Annotation,
  Bars,
  Curve,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  type AxisModel,
} from 'aifn-render/viz'
import { formatValue, TreeView } from '@lab/views'

// ── Data: noisy, so a fully grown tree overfits ──────────────────────────────────────────────────────────────────

const SETS = {
  moons: { label: 'two moons', make: (s: Stream) => moons(s, { n: 240, noise: 0.3 }) },
  circles: { label: 'two circles', make: (s: Stream) => circles(s, { n: 240, noise: 0.15, factor: 0.55 }) },
  xor: { label: 'XOR', make: (s: Stream) => xor(s, { n: 240, kind: 'gaussian', sd: 0.6 }) },
  blobs: {
    label: 'three blobs',
    make: (s: Stream) =>
      blobs(s, {
        n: 240,
        centers: [
          [-1.6, -1],
          [1.6, -1],
          [0, 1.6],
        ],
        sd: 1.1,
      }),
  },
} as const
type SetName = keyof typeof SETS
const SET_OPTIONS = (Object.keys(SETS) as SetName[]).map((value) => ({ value, label: SETS[value].label }))
const NOISE_OPTIONS = [
  { value: 0, label: 'none' },
  { value: 0.05, label: '5%' },
  { value: 0.1, label: '10%' },
  { value: 0.2, label: '20%' },
]
const GRID = 72

type Data = {
  train: Dataset
  valid: Dataset
  x: Tensor
  y: Tensor
  /** Training columns and labels. */
  x0: number[]
  x1: number[]
  labels: number[]
  names: readonly string[]
  lower: [number, number]
  upper: [number, number]
  field: { x: number[]; y: number[]; points: Tensor }
}

function makeData(name: SetName, noise: number): Data {
  const draw = (part: string) => {
    const d = SETS[name].make(stream(`showcase-trees/${name}/${part}`))
    return noise > 0 ? withLabelNoise(stream(`showcase-trees/${name}/${part}/noise`), d, { rate: noise }) : d
  }
  const train = draw('train')
  const valid = draw('valid')
  const xs = toFlat(train.x as Tensor)
  const vs = toFlat(valid.x as Tensor)
  const all = [...xs, ...vs]
  const col = (f: number) => all.filter((_, i) => i % 2 === f)
  const pad = (v: number[]): [number, number] => {
    const lo = Math.min(...v)
    const hi = Math.max(...v)
    return [lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)]
  }
  const [a, b] = [pad(col(0)), pad(col(1))]
  const gx = Array.from({ length: GRID }, (_, j) => a[0] + ((j + 0.5) * (a[1] - a[0])) / GRID)
  const gy = Array.from({ length: GRID }, (_, i) => b[0] + ((i + 0.5) * (b[1] - b[0])) / GRID)
  const points = fromData(Float64Array.from(gy.flatMap((y) => gx.flatMap((x) => [x, y]))), [GRID * GRID, 2])
  return {
    train,
    valid,
    x: train.x as Tensor,
    y: train.y as Tensor,
    x0: xs.filter((_, i) => i % 2 === 0),
    x1: xs.filter((_, i) => i % 2 === 1),
    labels: toFlat(train.y as Tensor),
    names: train.meta.labelNames ?? ['class 0', 'class 1'],
    lower: [a[0], b[0]],
    upper: [a[1], b[1]],
    field: { x: gx, y: gy, points },
  }
}

// ── Shared state: the data and the tree, and the reader's selection ─────────────────────────────────────────────

const ORDER_OPTIONS = [
  { value: 'depth-first', label: 'depth-first' },
  { value: 'breadth-first', label: 'breadth-first' },
  { value: 'best-first', label: 'best-first' },
] as const
const LEAF_BUDGETS = [
  { value: 0, label: 'none' },
  { value: 4, label: '4' },
  { value: 8, label: '8' },
  { value: 16, label: '16' },
  { value: 32, label: '32' },
]

/** The selection: nothing, a node (by id) or a query point, held in the figure state and so in the URL. */
const selection = (lower: readonly number[], upper: readonly number[]) => ({
  pick: choice(['none', 'node', 'point'], 'none', { onChart: true }),
  node: slider(0, 4000, 0, { step: 1, onChart: true }),
  qx: slider(lower[0], upper[0], (lower[0] + upper[0]) / 2, { step: 0.01, onChart: true }),
  qy: slider(lower[1], upper[1], (lower[1] + upper[1]) / 2, { step: 0.01, onChart: true }),
  boundary: toggle(true, 'decision boundaries'),
})
type SelectionState = FigureState<ReturnType<typeof selection>>

const CRITERION_TEX = {
  gini: String.raw`\Delta\mathrm{Gini}`,
  entropy: String.raw`\Delta H`,
  squared: String.raw`\Delta\mathrm{MSE}`,
}
const f3 = (v: number) => (Number.isFinite(v) ? Number(v.toPrecision(3)).toString() : '—')
const xName = (f: number) => `x_${f}`

const STOP_TEXT: Record<LeafReason, string> = {
  pure: 'it is pure',
  'max-depth': 'it is at the maximum depth',
  'min-samples': 'it has too few rows to split',
  'no-split': 'no threshold leaves enough rows on both sides',
  'min-decrease': 'no split decreases impurity enough',
  'max-leaves': 'the leaf budget is spent',
  pruned: 'pruning cut its subtree',
}

/** One sentence (with `$…$` maths) explaining the decision recorded at a node. */
function explainNode(tree: DecisionTree, id: number, names: readonly string[]): string {
  const node = tree.nodes[id]
  const d = CRITERION_TEX[tree.criterion]
  if (node.stop !== null) {
    const c = nodePrediction(tree, id)
    return `node ${id} is a leaf: ${STOP_TEXT[node.stop]}. It holds ${node.count} rows and predicts ${names[c] ?? `class ${c}`} (share ${f3(node.value[c])}).`
  }
  const ranked = [...node.splits].sort((a, b) => b.decrease - a.decrease)
  const parts = ranked.map((s) =>
    s.feature === node.feature
      ? `best on $${xName(s.feature)}$ gives $${d} = ${f3(s.decrease)}$`
      : `best on $${xName(s.feature)}$ (at ${f3(s.threshold)}) gives ${f3(s.decrease)}`,
  )
  return `split node ${id} on $${xName(node.feature)} \\le ${f3(node.threshold)}$: ${parts.join(' · ')} → $${xName(node.feature)}$ chosen.`
}

/** A point's decision path as text: each test and its outcome, then the leaf. */
function explainPath(tree: DecisionTree, path: DecisionPath, names: readonly string[]): string {
  const tests = path.tests.map(
    (t) => `$${xName(t.feature)} = ${f3(t.value)} \\le ${f3(t.threshold)}$? ${t.left ? 'yes → left' : 'no → right'}`,
  )
  const c = nodePrediction(tree, path.leaf)
  return `${tests.length ? tests.join(' · ') + ' · ' : ''}leaf ${path.leaf}: ${names[c] ?? `class ${c}`} (share ${f3(tree.nodes[path.leaf].value[c])})`
}

function Band({ children }: { children: string }) {
  return (
    <div className="font-prose text-[15px] leading-snug">
      <MathText text={children} />
    </div>
  )
}

/** What the selection points at in `tree`, among the nodes `within` allows. */
function useFocus(
  tree: DecisionTree,
  figure: SelectionState,
  within: (id: number) => boolean,
  fallback: number | null,
) {
  const { pick, node, qx, qy } = figure
  const path = useMemo(
    () => (pick === 'point' ? decisionPath(tree, [qx, qy], { within }) : null),
    [pick, qx, qy, tree, within],
  )
  const chosen = pick === 'node' && node < tree.nodes.length && within(node) ? node : null
  const focus = path ? path.leaf : (chosen ?? fallback)
  return { path, chosen, focus }
}

/** Click a node to pin it, again or Escape to unpin: the shared hover-and-pin helper over the figure's pick. */
function useNodePin(figure: SelectionState) {
  const pins = usePinned(figure.pick === 'node' ? figure.node : -1, (v) => {
    if (v < 0) figure.set('pick', 'none')
    else {
      figure.set('node', v)
      figure.set('pick', 'node')
    }
  })
  return pins.toggle
}

// ── The scatter: decision regions, the focus node's region, the query point ─────────────────────────────────────

function Scatter({
  data,
  tree,
  within,
  focus,
  path,
  figure,
  ax0,
  ax1,
  guide,
}: {
  data: Data
  tree: DecisionTree
  within: (id: number) => boolean
  focus: number | null
  path: DecisionPath | null
  figure: SelectionState
  ax0: AxisModel
  ax1: AxisModel
  /** A candidate split hovered on the criterion chart: drawn across the focus node's box along its feature. */
  guide?: { feature: number; threshold: number } | null
}) {
  const regions = useMemo(() => {
    const labels = toFlat(decideTree(tree, data.field.points, { within }))
    return Array.from({ length: GRID }, (_, i) => labels.slice(i * GRID, (i + 1) * GRID))
  }, [tree, data, within])
  const box = (id: number) => nodeRegion(tree, id, { lower: data.lower, upper: data.upper })
  // Points keep their true class's colour; the shape says whether the tree at this step classifies them correctly
  // (circle, shape 0) or not (square, shape 1).
  const wrong = useMemo(() => {
    const predicted = toFlat(decideTree(tree, data.x, { within }))
    return Array.from(data.labels, (y, i) => (predicted[i] === y ? 0 : 1))
  }, [tree, data, within])
  const inside = useMemo(() => {
    if (focus === null) return null
    const rows = tree.nodes[focus].rows
    return {
      x: Array.from(rows, (i) => data.x0[i]),
      y: Array.from(rows, (i) => data.x1[i]),
      group: Array.from(rows, (i) => data.labels[i]),
      shape: Array.from(rows, (i) => wrong[i]),
    }
  }, [focus, tree, data, wrong])
  const focusBox = focus === null ? null : box(focus)
  const rect = (b: { lower: number[]; upper: number[] }) => {
    const [x0, y0] = b.lower
    const [x1, y1] = b.upper
    return [
      { from: [x0, y0], to: [x1, y0] },
      { from: [x1, y0], to: [x1, y1] },
      { from: [x1, y1], to: [x0, y1] },
      { from: [x0, y1], to: [x0, y0] },
    ] as const
  }
  // Every region on the way to the selected node (a query point's leaf, or a clicked node), from the root's child down:
  // each ancestor's box is outlined, thicker the deeper it is, so the nested half-spaces that decided the point read as
  // layers, the selected node's own box the thickest.
  const chain = useMemo(() => {
    const target = path ? path.leaf : focus
    if (target === null) return []
    const out: number[] = []
    for (let v: number | null = target; v !== null; v = tree.nodes[v].parent) out.unshift(v)
    return out.slice(1) // the root's box is the whole plot
  }, [path, focus, tree])
  const layers = chain.map((v, k) => ({
    id: v,
    width: 1 + (2.5 * (k + 1)) / chain.length,
    last: k === chain.length - 1,
  }))
  const select = (p: [number, number]) => {
    figure.set('qx', p[0])
    figure.set('qy', p[1])
    figure.set('pick', 'point')
  }
  return (
    <Plot x={ax0} y={ax1} onPlotClick={figure.pick === 'point' ? undefined : select}>
      <Raster
        x={data.field.x}
        y={data.field.y}
        z={regions}
        scale="categorical"
        fillOpacity={0.22}
        categoryNames={data.names}
        boundary={figure.boundary}
      />
      {focusBox && (
        <Bars
          name="selected region"
          x={[(focusBox.lower[0] + focusBox.upper[0]) / 2]}
          y={[focusBox.upper[1]]}
          edges={[focusBox.lower[0], focusBox.upper[0]]}
          base={focusBox.lower[1]}
          slot={nodePrediction(tree, focus!)}
          opacity={0.22}
        />
      )}
      {guide && focusBox && (
        <Segments
          segments={[
            guide.feature === 0
              ? { from: [guide.threshold, focusBox.lower[1]], to: [guide.threshold, focusBox.upper[1]] }
              : { from: [focusBox.lower[0], guide.threshold], to: [focusBox.upper[0], guide.threshold] },
          ]}
          emphasis
          width={2}
          dashed
          live
        />
      )}
      {layers.map((l) => (
        <Segments key={l.id} segments={rect(box(l.id))} emphasis={l.last} width={l.width} dashed={!l.last} />
      ))}
      {inside ? (
        <>
          {/* Outside the selected node: still in their class colours, smaller and lighter. */}
          <Points name="elsewhere" x={data.x0} y={data.x1} group={data.labels} shape={wrong} thin />
          <Points name="rows" {...inside} groupNames={data.names} />
        </>
      ) : (
        <Points name="rows" x={data.x0} y={data.x1} group={data.labels} shape={wrong} groupNames={data.names} />
      )}
      {figure.pick === 'point' && <Handle {...figure.handle(['qx', 'qy'], { label: 'query' })} />}
    </Plot>
  )
}

// ── The split criterion at the focus node ────────────────────────────────────────────────────────────────────────

function SplitPanel({
  data,
  tree,
  focus,
  axes,
  onHover,
}: {
  data: Data
  tree: DecisionTree
  focus: number | null
  /** The threshold under the pointer on the criterion chart (null when it leaves), for the scatter's guide line. */
  onHover?: (t: { feature: number; threshold: number } | null) => void
  axes: {
    threshold: AxisModel
    decrease: AxisModel
    feature: AxisModel
    best: AxisModel
    cls: AxisModel
    count: AxisModel
  }
}) {
  const node = focus === null ? null : tree.nodes[focus]
  const split = node && node.stop === null ? node : null
  const curve = useMemo(
    () => (split ? splitCurve(tree, split.id, split.feature, { x: data.x, y: data.y }) : null),
    [split, tree, data],
  )
  if (!node) return <Empty>Click a node, or a point on the scatter.</Empty>
  if (split && curve) {
    const chosen = split.splits.find((s) => s.feature === split.feature)!
    const others = split.splits.filter((s) => s.feature !== split.feature && Number.isFinite(s.decrease))
    return (
      <Dashboard>
        <DashboardRow>
          <DashboardCell ratio={2.2}>
            <Plot
              x={axes.threshold}
              y={axes.decrease}
              legend={false}
              onPointer={(e) => onHover?.(e.type === 'move' ? { feature: split.feature, threshold: e.point[0] } : null)}
            >
              <Curve
                name={`decrease on x${'₀₁'[split.feature]}`}
                x={toFlat(curve.thresholds)}
                y={toFlat(curve.decreases)}
                emphasis
              />
              <Annotation x={split.threshold} text={`t = ${f3(split.threshold)}`} dashed />
              <Points name="chosen" x={[split.threshold]} y={[chosen.decrease]} emphasis />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={axes.feature} y={axes.best} legend={false}>
              <Bars name="chosen" x={[chosen.feature]} y={[chosen.decrease]} emphasis />
              {others.length > 0 && (
                <Bars
                  name="best of the others"
                  x={others.map((s) => s.feature)}
                  y={others.map((s) => s.decrease)}
                  muted
                />
              )}
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    )
  }
  const counts = node.counts
  return (
    <Plot x={axes.cls} y={axes.count}>
      {counts.map((c, k) => (
        <Bars key={k} name={data.names[k] ?? `class ${k}`} x={[k]} y={[c]} slot={k} />
      ))}
    </Plot>
  )
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-muted-foreground">{children}</div>
}

function useSplitAxes(key: unknown, classes: readonly string[]) {
  return {
    threshold: useAxis({ label: 'threshold' }),
    decrease: useAxis({ label: 'impurity decrease' }),
    feature: useAxis({ label: 'feature', categories: ['x₀', 'x₁'] }),
    best: useAxis({ label: 'best decrease', range: [0, undefined] }),
    cls: useAxis({ label: 'class', categories: classes, key }),
    count: useAxis({ label: 'rows', range: [0, undefined] }),
  }
}

function NodeReadouts({
  tree,
  focus,
  path,
  names,
}: {
  tree: DecisionTree
  focus: number | null
  path: DecisionPath | null
  names: readonly string[]
}) {
  if (focus === null) return null
  const node = tree.nodes[focus]
  const c = nodePrediction(tree, focus)
  return (
    <>
      <Readout label="node" value={`${focus} (depth ${node.depth})`} />
      <Readout label="rows" value={node.count} />
      <Readout label="class counts" value={node.counts.map((v) => formatValue(v)).join(' / ')} />
      <Readout label={`impurity (${tree.criterion})`} value={formatValue(node.impurity)} />
      <Readout
        label="rule"
        value={node.stop === null ? `x${'₀₁'[node.feature]} ≤ ${f3(node.threshold)}` : `leaf: ${node.stop}`}
      />
      {path && <Readout label="prediction" value={`${names[c] ?? c} (${f3(node.value[c])})`} />}
    </>
  )
}

/** Edge labels on a decision path: yes (≤, left) or no (>, right). */
const pathLabels = (tree: DecisionTree, path: DecisionPath | null) => (child: number) => {
  if (!path || !path.nodes.includes(child)) return undefined
  return tree.nodes[child].slot === 0 ? 'yes' : 'no'
}

// ── Figure 1: grow and inspect ───────────────────────────────────────────────────────────────────────────────────

function GrowFigure() {
  const [guide, setGuideState] = useState<{ feature: number; threshold: number } | null>(null)
  // Only a changed guide re-renders: pointer moves at the same place, or repeated leaves, are ignored.
  const setGuide = (g: { feature: number; threshold: number } | null) =>
    setGuideState((old) => (old?.feature === g?.feature && old?.threshold === g?.threshold ? old : g))
  const [step, setStep] = useState(0)
  const base = useFigureState({
    data: row('1 · data', {
      dataset: choice(SET_OPTIONS, 'moons', { label: 'dataset' }),
      noise: choice(NOISE_OPTIONS, 0.1, { label: 'label noise' }),
    }),
    tree: row('2 · tree', {
      criterion: choice(['gini', 'entropy'], 'gini', { label: 'impurity' }),
      order: choice(ORDER_OPTIONS, 'depth-first', { label: 'expansion order' }),
      depth: slider(1, 12, 6, { label: 'maximum depth', step: 1 }),
      minLeaf: slider(1, 20, 3, { label: 'min samples per leaf', step: 1 }),
      maxLeaves: choice(LEAF_BUDGETS, 0, { label: 'leaf budget' }),
    }),
    ...selection([-6, -6], [6, 6]),
  })
  const figure = base as unknown as typeof base & SelectionState
  const name = figure.data.dataset as SetName
  const data = useMemo(() => makeData(name, figure.data.noise as number), [name, figure.data.noise])
  const { criterion, order, depth, minLeaf, maxLeaves } = figure.tree
  const model = useMemo(
    () =>
      decisionTree({
        criterion: criterion as 'gini' | 'entropy',
        order: order as GrowthOrder,
        maxDepth: depth,
        minSamplesLeaf: minLeaf,
        ...(maxLeaves ? { maxLeaves: maxLeaves as number } : {}),
      }).fit(toDataset(data.x, data.y)),
    [data, criterion, order, depth, minLeaf, maxLeaves],
  )
  const tree = model.tree
  const steps = model.training.steps
  const k = Math.min(step, steps.length - 2)
  const state = steps[k + 1]
  const created = state.tree.nodes.length
  const within = useMemo(() => (v: number) => v < created, [created])
  const { path, chosen, focus } = useFocus(tree, figure, within, state.current)
  const ax0 = useAxis({ label: 'x₀', hold: 'initial', key: name })
  const ax1 = useAxis({ label: 'x₁', hold: 'initial', key: name, equal: ax0 })
  const axes = useSplitAxes(name, data.names)
  const select = useNodePin(figure)
  const band =
    path !== null
      ? explainPath(tree, path, data.names)
      : chosen !== null
        ? explainNode(tree, chosen, data.names)
        : `step ${k + 1}: ${explainNode(state.tree, state.current, data.names)}`
  return (
    <Figure
      title="Growing a tree, and reading it"
      purpose="Each step creates one node: the fitter searches every threshold of every feature, splits on the largest impurity decrease, or records why the node stops."
      state={figure}
      defaultSize="XL"
      controls={
        <Player value={k} onChange={setStep} count={steps.length - 1} label="node" format={(p) => String(p + 1)} />
      }
      equation={<Band>{band}</Band>}
      readouts={{
        tree: (
          <>
            <Readout label="nodes" value={`${created} of ${tree.nodes.length}`} />
            <Readout label="leaves (final)" value={tree.nodes.filter((n) => !n.children.length).length} />
            <Readout label="order" value={order} />
          </>
        ),
        [focus === state.current && chosen === null && !path ? 'this step' : 'selected']: (
          <NodeReadouts tree={tree} focus={focus} path={path} names={data.names} />
        ),
      }}
      caption={
        <>
          Points keep their true class&apos;s colour; with decision boundaries on, ink lines separate the predicted
          classes; circles are classified correctly by the tree at this step and squares wrongly. Play the growth one
          node at a time. The band explains the step: every feature&apos;s best threshold and its decrease, the largest
          taken, or why a node stops. Click a node (or use Tab and the arrow keys) to see its region, the intersection
          of its ancestors&apos; half-spaces, and its split criterion; click it again or press Escape to unpin. Click
          the scatter to drop a query point (then drag it) and follow its decision path. Without a leaf budget every
          expansion order ends in the same tree, since each split depends only on its own node&apos;s rows; with a
          budget, best-first spends it on the splits that decrease impurity most.
        </>
      }
    >
      <Dashboard>
        <DashboardRow fit="width" minHeight={260}>
          <DashboardCell>
            <TreeView
              tree={tree}
              ariaLabel="The tree being grown"
              hidden={(v) => v >= created}
              nodeState={(v) => (v === state.current && chosen === null && !path ? 'active' : undefined)}
              shape={(v) => (tree.nodes[v].children.length && v < created ? 'box' : 'pill')}
              selected={chosen ?? (path ? path.leaf : null)}
              onSelect={select}
              path={path?.nodes}
              edgeLabels={pathLabels(tree, path)}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={240}>
          <DashboardCell ratio={1.2}>
            <Scatter
              data={data}
              tree={tree}
              within={within}
              focus={focus}
              path={path}
              figure={figure}
              ax0={ax0}
              ax1={ax1}
              guide={guide}
            />
          </DashboardCell>
          <DashboardCell ratio={1.4}>
            <SplitPanel data={data} tree={tree} focus={focus} axes={axes} onHover={setGuide} />
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ── Figure 2: prune ──────────────────────────────────────────────────────────────────────────────────────────────

function PruneFigure() {
  const [guide, setGuideState] = useState<{ feature: number; threshold: number } | null>(null)
  // Only a changed guide re-renders: pointer moves at the same place, or repeated leaves, are ignored.
  const setGuide = (g: { feature: number; threshold: number } | null) =>
    setGuideState((old) => (old?.feature === g?.feature && old?.threshold === g?.threshold ? old : g))
  const [step, setStep] = useState(0)
  const base = useFigureState({
    data: row('1 · data', {
      dataset: choice(SET_OPTIONS, 'moons', { label: 'dataset' }),
      noise: choice(NOISE_OPTIONS, 0.1, { label: 'label noise' }),
      criterion: choice(['gini', 'entropy'], 'gini', { label: 'impurity' }),
    }),
    ...selection([-6, -6], [6, 6]),
  })
  const figure = base as unknown as typeof base & SelectionState
  const name = figure.data.dataset as SetName
  const data = useMemo(() => makeData(name, figure.data.noise as number), [name, figure.data.noise])
  const criterion = figure.data.criterion as 'gini' | 'entropy'
  const tree = useMemo(() => decisionTree({ criterion }).fit(toDataset(data.x, data.y)).tree, [data, criterion])
  const ccp = useMemo(() => {
    const p = costComplexityPath(tree)
    return { alphas: toFlat(p.alphas), leaves: toFlat(p.leaves), cuts: toFlat(p.cuts) }
  }, [tree])
  const last = ccp.cuts.length
  const k = Math.min(step, last)
  // Train and validation error of the tree after each cut, read in place in the full tree's ids.
  const errors = useMemo(() => {
    const train: number[] = []
    const valid: number[] = []
    for (let s = 0; s <= last; s++) {
      const kept = keptNodes(tree, ccp.cuts.slice(0, s))
      const within = (v: number) => kept[v] === 1
      train.push(errorRate(data.y, decideTree(tree, data.x, { within })))
      valid.push(errorRate(data.valid.y as Tensor, decideTree(tree, data.valid.x as Tensor, { within })))
    }
    let best = 0
    valid.forEach((e, s) => {
      if (e < valid[best] - 1e-12) best = s
    })
    return { train, valid, best }
  }, [tree, ccp, last, data])
  const cut = useMemo(() => ccp.cuts.slice(0, k), [ccp, k])
  const kept = useMemo(() => keptNodes(tree, cut), [tree, cut])
  const next = k < last ? ccp.cuts[k] : null
  const within = useMemo(() => (v: number) => kept[v] === 1, [kept])
  const nextSubtree = useMemo(() => {
    if (next === null) return new Set<number>()
    const out = new Set<number>()
    const stack = [next]
    while (stack.length) {
      const v = stack.pop()!
      out.add(v)
      if (within(v)) stack.push(...tree.nodes[v].children.filter(within))
    }
    return out
  }, [next, tree, within])
  const { path, chosen, focus } = useFocus(tree, figure, within, next)
  const ax0 = useAxis({ label: 'x₀', hold: 'initial', key: name })
  const ax1 = useAxis({ label: 'x₁', hold: 'initial', key: name, equal: ax0 })
  const axes = useSplitAxes(name, data.names)
  // The last cut (to the root) sits far right of the rest; the axis ends at the cut before it.
  const top = (ccp.alphas[Math.max(last - 1, 1)] || ccp.alphas[last]) * 1.15 || 0.1
  const alphaAxis = useAxis({ label: 'α', range: [0, top], key: `${name}/${criterion}` })
  const errAxis = useAxis({ label: 'error rate', range: [0, undefined], hold: 'initial', key: `${name}/${criterion}` })
  // Error is constant between breakpoints: each value holds from its α to the next.
  const curves = useMemo(() => {
    const x = ccp.alphas.flatMap((a, s) => [a, s < last ? ccp.alphas[s + 1] : top])
    const hold = (ys: number[]) => ys.flatMap((e) => [e, e])
    return { x, train: hold(errors.train), valid: hold(errors.valid) }
  }, [ccp, last, top, errors])
  const alpha = ccp.alphas[k]
  const select = useNodePin(figure)
  const onAlpha = (a: number) =>
    setStep(
      Math.max(
        0,
        ccp.alphas.findLastIndex((b) => b <= a + 1e-12),
      ),
    )
  const band =
    path !== null
      ? explainPath(tree, path, data.names)
      : chosen !== null
        ? explainNode(tree, chosen, data.names)
        : next !== null
          ? `step ${k}: next, at $\\alpha = ${f3(ccp.alphas[k + 1])}$, the weakest link is node ${next}: collapsing its ${
              [...nextSubtree].filter((v) => !tree.nodes[v].children.some(within)).length
            } leaves into one costs the least training impurity per leaf removed.`
          : `step ${k}: the tree is pruned to its root.`
  return (
    <Figure
      title="Pruning it back"
      purpose="Cost-complexity pruning cuts the weakest link at each step (the subtree that buys least impurity per leaf), and validation error says where to stop."
      state={figure}
      defaultSize="XL"
      controls={
        <Player
          value={k}
          onChange={setStep}
          count={last + 1}
          label="cut"
          format={(p) => `${p} (α ${f3(ccp.alphas[p])})`}
        />
      }
      equation={<Band>{band}</Band>}
      readouts={{
        pruning: (
          <>
            <Readout label="α" value={f3(alpha)} />
            <Readout label="leaves" value={ccp.leaves[k]} />
            <Readout label="full tree leaves" value={ccp.leaves[0]} />
            <Readout label="training error" value={f3(errors.train[k])} />
            <Readout label="validation error" value={f3(errors.valid[k])} />
            <Readout
              label="best α (validation)"
              value={`${f3(ccp.alphas[errors.best])} (${ccp.leaves[errors.best]} leaves)`}
            />
          </>
        ),
        selected: <NodeReadouts tree={tree} focus={focus} path={path} names={data.names} />,
      }}
      caption={
        <>
          Step 0 is the fully grown tree, which fits the noisy labels: training error 0. Each step collapses the
          emphasised subtree (its region outlined on the scatter) into a leaf; the α line moves to that breakpoint and
          can be dragged. Training error rises with every cut, while validation error first falls: the marked α is the
          best on held-out data. Circles and squares mark the training points the pruned tree gets right and wrong.
          Click a node or the scatter, as above.
        </>
      }
    >
      <Dashboard>
        <DashboardRow fit="width" minHeight={260}>
          <DashboardCell>
            <TreeView
              tree={tree}
              ariaLabel="The tree being pruned"
              cut={cut}
              nodeState={(v) => (nextSubtree.has(v) && chosen === null && !path ? 'active' : undefined)}
              edgeState={(c) => (nextSubtree.has(c) && c !== next && chosen === null && !path ? 'active' : undefined)}
              shape={(v) => (tree.nodes[v].children.length && !cut.includes(v) ? 'box' : 'pill')}
              selected={chosen ?? (path ? path.leaf : null)}
              onSelect={select}
              path={path?.nodes}
              edgeLabels={pathLabels(tree, path)}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={240}>
          <DashboardCell>
            <Scatter
              data={data}
              tree={tree}
              within={within}
              focus={focus}
              path={path}
              figure={figure}
              ax0={ax0}
              ax1={ax1}
              guide={guide}
            />
          </DashboardCell>
          <DashboardCell>
            <Plot x={alphaAxis} y={errAxis}>
              <Curve name="training error" x={curves.x} y={curves.train} muted />
              <Curve name="validation error" x={curves.x} y={curves.valid} emphasis />
              <Annotation at={[ccp.alphas[errors.best], errors.valid[errors.best]]} text="best" />
              <Handle kind="x" at={alpha} onDrag={onAlpha} label={`α = ${f3(alpha)}`} />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <SplitPanel data={data} tree={tree} focus={focus} axes={axes} onHover={setGuide} />
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** The showcase: grow (and inspect), then prune. */
export function DecisionTreeShowcase() {
  return (
    <>
      <GrowFigure />
      <PruneFigure />
    </>
  )
}

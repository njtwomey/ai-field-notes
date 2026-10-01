import { useMemo, useState } from 'react'
import { dataset as toDataset } from 'aifn/learning/estimators'
import { adaBoost, gradientBoosting } from 'aifn-applied/learning/trees-and-ensembles/boosting'
import { costComplexityPath, decisionTree, pruneTree, treeSize } from 'aifn-applied/learning/trees-and-ensembles'
import { grid2d } from 'aifn/numerics/geometry'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Player } from '@lab/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Contours, Curve, Handle, Plot, Points, Raster, Readout, useAxis } from '@lab/viz'
import { formatValue, TreeView } from '@lab/views'
import { DATASET_OPTIONS, DATASETS, type DatasetName } from './data'

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

/** z[i][j] from a flat [ny · nx] vector. */
const reshape = (v: number[], ny: number, nx: number) =>
  Array.from({ length: ny }, (_, i) => v.slice(i * nx, (i + 1) * nx))

// ── Tree growth and the split search ─────────────────────────────────────────────────────────────────────────────

export function TreeGrowthSpecimen() {
  const figure = useFigureState({
    setup: row('1 · data and criterion', {
      dataset: choice(DATASET_OPTIONS, 'blobs', { label: 'dataset' }),
      criterion: choice(['gini', 'entropy'], 'gini', { label: 'impurity' }),
      depth: slider(1, 6, 4, { label: 'maximum depth', step: 1 }),
    }),
  })
  const dataset = figure.setup.dataset as DatasetName
  const criterion = figure.setup.criterion as 'gini' | 'entropy'
  const { depth } = figure.setup
  const [step, setStep] = useState(0)
  const data = useMemo(() => DATASETS[dataset].make(), [dataset])
  const model = useMemo(
    () => decisionTree({ criterion, maxDepth: depth }).fit(toDataset(data.x, data.y!)),
    [data, criterion, depth],
  )
  const t = model.training
  const k = Math.max(1, Math.min(step, t.steps.length - 1))
  const state = t.steps[k]
  const created = state.tree.nodes.length
  const cols = useMemo(() => columns(data.x), [data])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const rows = useMemo(() => toFlat(state.rows), [state])
  const search = state.search
  const splits = useMemo(
    () =>
      (search?.candidates ?? []).map((c) => ({
        name: `split on x${'₀₁'[c.feature]}`,
        x: toFlat(c.thresholds),
        y: toFlat(c.decreases),
        slot: c.feature,
      })),
    [search],
  )
  const here = useMemo(
    () => ({
      x: rows.map((i) => cols.x0[i]),
      y: rows.map((i) => cols.x1[i]),
      group: rows.map((i) => labels[i]),
    }),
    [rows, cols, labels],
  )
  const threshold = useAxis({ label: 'threshold' })
  const decrease = useAxis({ label: 'impurity decrease' })
  const ax0 = useAxis({ label: 'x₀', hold: 'initial', key: dataset })
  const ax1 = useAxis({ label: 'x₁', hold: 'initial', key: dataset })
  const node = state.tree.nodes[state.current]
  return (
    <Figure
      title="Growing a CART tree, node by node"
      purpose="Depth-first growth: each step creates one node, searches every threshold of every feature for the largest weighted impurity decrease, and either splits or stops."
      state={figure}
      defaultSize="XL"
      controls={
        <ControlRow label="2 · growth">
          <Player value={step} onChange={setStep} count={t.steps.length} label="node" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="node" value={`${state.current} (depth ${node.depth})`} />
          <Readout label="rows at the node" value={node.count} />
          <Readout label="impurity" value={formatValue(node.impurity)} />
          <Readout
            label="split"
            value={node.feature >= 0 ? `x${'₀₁'[node.feature]} ≤ ${formatValue(node.threshold)}` : 'leaf'}
          />
          <Readout label="decrease" value={formatValue(node.decrease)} />
        </>
      }
      caption="Nodes appear in the order they are created (preorder, left first); the current node is emphasised. The lower left panel is the split search at that node: the impurity decrease of every candidate threshold, one curve per feature, whose maximum is the split taken. The lower right panel colours the rows that reached the node by class and greys the rest."
    >
      <Dashboard>
        <DashboardRow minHeight={220}>
          <DashboardCell>
            <TreeView
              tree={model.tree}
              ariaLabel="The tree being grown"
              hidden={(v) => v >= created}
              nodeState={(v) => (v === state.current ? 'active' : 'done')}
              shape={(v) => (model.tree.nodes[v].children.length ? 'box' : 'pill')}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={220}>
          <DashboardCell>
            <Plot x={threshold} y={decrease}>
              {splits.map((c) => (
                <Curve key={c.name} name={c.name} x={c.x} y={c.y} slot={c.slot} />
              ))}
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={ax0} y={ax1}>
              <Points name="elsewhere" x={cols.x0} y={cols.x1} muted />
              <Points name="at this node" {...here} groupNames={data.meta.labelNames} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ── Cost-complexity pruning ──────────────────────────────────────────────────────────────────────────────────────

export function PruningSpecimen() {
  const data = useMemo(() => DATASETS.moons.make(), [])
  const full = useMemo(() => decisionTree().fit(toDataset(data.x, data.y!)).tree, [data])
  const path = useMemo(() => costComplexityPath(full), [full])
  const alphas = useMemo(() => toFlat(path.alphas), [path])
  const impurities = useMemo(() => toFlat(path.impurities), [path])
  const leaves = toFlat(path.leaves)
  const top = alphas[alphas.length - 2] ?? 0.1
  const figure = useFigureState({ alpha: slider(0, top * 1.1, 0.01, { label: 'α', onChart: true }) })
  const { alpha } = figure
  const pruned = useMemo(() => pruneTree(full, alpha), [full, alpha])
  const size = treeSize(pruned)
  const steps = useMemo(
    () => ({
      x: alphas.slice(0, -1).flatMap((a, i) => [a, alphas[i + 1]]),
      y: impurities.slice(0, -1).flatMap((r) => [r, r]),
    }),
    [alphas, impurities],
  )
  const alphaAxis = useAxis({ label: 'α', range: [0, top * 1.2] })
  const rAxis = useAxis({ label: 'total leaf impurity R' })
  return (
    <Figure
      title="Minimal cost-complexity pruning"
      purpose="Pruning at complexity α keeps the subtree minimising R(T) + α|leaves(T)|; the weakest links go first, at the α printed on the path."
      state={figure}
      defaultSize="XL"
      readouts={
        <>
          <Readout label="α" value={formatValue(alpha)} />
          <Readout label="R(T_α)" value={formatValue(impurities[alphas.findLastIndex((a) => a <= alpha)] ?? NaN)} />
          <Readout label="leaves on the path" value={leaves[alphas.findLastIndex((a) => a <= alpha)] ?? '—'} />
          <Readout label="leaves" value={size.leaves} />
          <Readout label="depth" value={size.depth} />
          <Readout label="full tree leaves" value={treeSize(full).leaves} />
        </>
      }
      caption="Drag the α line on the path: the total leaf impurity R rises in steps as each weakest link is cut, and the tree above shrinks to match. At α = 0 the tree is the fully grown one, with pure leaves."
    >
      <Dashboard>
        <DashboardRow ratio={1.5} minHeight={240}>
          <DashboardCell>
            <TreeView
              tree={pruned}
              ariaLabel="The pruned tree"
              shape={(v) => (pruned.nodes[v].children.length ? 'box' : 'pill')}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={180}>
          <DashboardCell>
            <Plot x={alphaAxis} y={rAxis} legend={false}>
              <Curve name="R(T_α)" x={steps.x} y={steps.y} />
              <Points name="pruning points" x={alphas} y={impurities} muted />
              <Handle {...figure.handle('alpha', { label: 'α' })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ── Boosting round by round ──────────────────────────────────────────────────────────────────────────────────────

export function BoostingSpecimen() {
  const figure = useFigureState({
    setup: row('1 · setup', {
      method: choice(
        [
          { value: 'adaboost', label: 'AdaBoost (SAMME)' },
          { value: 'gradient', label: 'gradient boosting (logistic)' },
        ],
        'adaboost',
        { label: 'method' },
      ),
      dataset: choice(
        DATASET_OPTIONS.filter((o) => o.value !== 'blobs'),
        'circles',
        { label: 'dataset' },
      ),
    }),
  })
  const method = figure.setup.method as 'adaboost' | 'gradient'
  const dataset = figure.setup.dataset as DatasetName
  const [round, setRound] = useState(0)
  const data = useMemo(() => DATASETS[dataset].make(), [dataset])
  const rounds = 80
  const ada = useMemo(() => adaBoost({ rounds }).fit(toDataset(data.x, data.y!)), [data])
  const gb = useMemo(
    () =>
      gradientBoosting({ loss: 'logistic', stages: rounds, learningRate: 0.3, tree: { maxDepth: 1 } }).fit(
        toDataset(data.x, data.y!),
      ),
    [data],
  )
  const g = useMemo(() => {
    const c = columns(data.x)
    const pad = (v: number[]) => {
      const lo = Math.min(...v)
      const hi = Math.max(...v)
      return [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)] as [number, number]
    }
    return grid2d(pad(c.x0), pad(c.x1), 70)
  }, [data])
  const K = ada.classes
  const r = Math.max(1, Math.min(round, method === 'adaboost' ? ada.learners.length : rounds))
  const z = useMemo(() => {
    const [ny, nx] = g.shape
    if (method === 'adaboost') {
      // The ensemble's vote share for class 1 (binary) or the winning class's share.
      const v = toRows(ada.votesUpTo(g.points, r))
      return reshape(
        v.map((row) => (K === 2 ? row[1] - row[0] : Math.max(...row))),
        ny,
        nx,
      )
    }
    const f = toFlat(gb.rawUpTo(g.points, r))
    return reshape(K === 2 ? f : f, ny, nx)
  }, [method, ada, gb, g, r, K])
  const state = method === 'adaboost' ? ada.training.steps[Math.min(r, ada.training.steps.length - 1)] : null
  const cols = useMemo(() => columns(data.x), [data])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const heavy = useMemo(() => {
    if (!state) return null
    const idx = toFlat(state.sampleWeights)
      .map((w, i) => [w, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, 15)
      .map((p) => p[1])
    return { x: idx.map((i) => cols.x0[i]), y: idx.map((i) => cols.x1[i]) }
  }, [state, cols])
  const field = useMemo(() => ({ x: toFlat(g.x), y: toFlat(g.y) }), [g])
  const ax0 = useAxis({ label: 'x₀', hold: 'initial', key: dataset })
  const ax1 = useAxis({ label: 'x₁', hold: 'initial', key: dataset, equal: ax0 })
  const loss = method === 'adaboost' ? toFlat(ada.training.series.trainingError) : toFlat(gb.training.series.loss)
  return (
    <Figure
      title="Boosting, round by round"
      purpose="Each round adds one stump fitted to what the ensemble so far gets wrong: reweighted rows for AdaBoost, pseudo-residuals for gradient boosting."
      state={figure}
      defaultSize="L"
      controls={
        <ControlRow label="2 · rounds">
          <Player value={round} onChange={setRound} count={rounds + 1} label="round" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="round" value={r} />
          <Readout
            label={method === 'adaboost' ? 'training error' : 'training log loss'}
            value={formatValue(loss[Math.min(r, loss.length - 1)])}
          />
          {state && <Readout label="this stump's α" value={formatValue(state.alphas[state.alphas.length - 1])} />}
        </>
      }
      caption="The colour is the ensemble's score after the chosen round (vote margin for AdaBoost, log-odds for gradient boosting), with its zero contour (ink) as the boundary. Round 0 shows the first stump. For AdaBoost, the ringed rows carry the most weight going into the next round: they sit where the current boundary is wrong. One stump draws one axis-aligned cut; eighty of them trace the circle."
    >
      <Plot x={ax0} y={ax1}>
        <Raster
          x={field.x}
          y={field.y}
          z={z}
          scale="diverging"
          range={method === 'adaboost' ? [-1, 1] : [-6, 6]}
          valueLabel="score"
        />
        <Contours x={field.x} y={field.y} z={z} levels={[0]} />
        <Points name="rows" x={cols.x0} y={cols.x1} group={labels} groupNames={data.meta.labelNames} />
        {heavy && <Points name="heaviest weights" x={heavy.x} y={heavy.y} emphasis />}
      </Plot>
    </Figure>
  )
}

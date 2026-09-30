import { useMemo, useState } from 'react'
import { dataset as toDataset } from 'aifn/learning/estimators'
import { adaBoost, gradientBoosting } from 'aifn-applied/learning/trees-and-ensembles/boosting'
import { costComplexityPath, decisionTree, pruneTree, treeSize } from 'aifn-applied/learning/trees-and-ensembles'
import { grid2d } from 'aifn/numerics/geometry'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { ChartSize, Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'
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
  const [dataset, setDataset] = useState<DatasetName>('blobs')
  const [criterion, setCriterion] = useState<'gini' | 'entropy'>('gini')
  const [depth, setDepth] = useState(4)
  const [step, setStep] = useState(8)
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
  const rows = toFlat(state.rows)
  const search = state.search
  const splitSeries = useMemo((): XYSeries[] => {
    if (!search) return []
    return search.candidates.map((c) => ({
      name: `split on x${'₀₁'[c.feature]}`,
      type: 'line' as const,
      x: toFlat(c.thresholds),
      y: toFlat(c.decreases),
      slot: c.feature,
    }))
  }, [search])
  const node = state.tree.nodes[state.current]
  return (
    <Figure
      title="Growing a CART tree, node by node"
      description="Depth-first growth: each step creates one node, searches every threshold of every feature for the largest weighted impurity decrease, and either splits or stops."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · data and criterion">
            <Select label="dataset" value={dataset} onChange={setDataset} options={DATASET_OPTIONS} />
            <Select label="impurity" value={criterion} onChange={setCriterion} options={['gini', 'entropy']} />
            <Slider label="maximum depth" value={depth} min={1} max={6} step={1} onChange={setDepth} />
          </ControlRow>
          <ControlRow label="2 · growth">
            <Player value={k} onChange={setStep} count={t.steps.length} label="node" />
          </ControlRow>
        </>
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
      <div className="flex flex-col gap-2">
        <ChartSize scale={0.5}>
          <TreeView
            tree={model.tree}
            ariaLabel="The tree being grown"
            hidden={(v) => v >= created}
            nodeState={(v) => (v === state.current ? 'active' : 'done')}
            shape={(v) => (model.tree.nodes[v].children.length ? 'box' : 'pill')}
          />
        </ChartSize>
        <ChartSize scale={0.5}>
          <Subplots cols={2}>
            <Panel>
              <XYChart series={splitSeries} xLabel="threshold" yLabel="impurity decrease" />
            </Panel>
            <Panel>
              <XYChart
                xLabel="x₀"
                yLabel="x₁"
                series={[
                  { name: 'elsewhere', type: 'scatter', x: cols.x0, y: cols.x1, muted: true },
                  {
                    name: 'at this node',
                    type: 'scatter',
                    x: rows.map((i) => cols.x0[i]),
                    y: rows.map((i) => cols.x1[i]),
                    group: rows.map((i) => labels[i]),
                    groupNames: data.meta.labelNames,
                  },
                ]}
              />
            </Panel>
          </Subplots>
        </ChartSize>
      </div>
    </Figure>
  )
}

// ── Cost-complexity pruning ──────────────────────────────────────────────────────────────────────────────────────

export function PruningSpecimen() {
  const data = useMemo(() => DATASETS.moons.make(), [])
  const full = useMemo(() => decisionTree().fit(toDataset(data.x, data.y!)).tree, [data])
  const path = useMemo(() => costComplexityPath(full), [full])
  const alphas = toFlat(path.alphas)
  const impurities = toFlat(path.impurities)
  const leaves = toFlat(path.leaves)
  const [alpha, setAlpha] = useState(0.01)
  const pruned = useMemo(() => pruneTree(full, alpha), [full, alpha])
  const size = treeSize(pruned)
  const top = alphas[alphas.length - 2] ?? 0.1
  return (
    <Figure
      title="Minimal cost-complexity pruning"
      description="Pruning at complexity α keeps the subtree minimising R(T) + α|leaves(T)|; the weakest links go first, at the α printed on the path."
      defaultSize="XL"
      controls={<Slider label="α" value={alpha} min={0} max={top * 1.1} onChange={setAlpha} />}
      readouts={
        <>
          <Readout label="leaves" value={size.leaves} />
          <Readout label="depth" value={size.depth} />
          <Readout label="full tree leaves" value={treeSize(full).leaves} />
        </>
      }
      caption="Drag the α line on the path: the total leaf impurity R rises in steps as each weakest link is cut, and the tree above shrinks to match. At α = 0 the tree is the fully grown one, with pure leaves."
    >
      <div className="flex flex-col gap-2">
        <ChartSize scale={0.6}>
          <TreeView
            tree={pruned}
            ariaLabel="The pruned tree"
            shape={(v) => (pruned.nodes[v].children.length ? 'box' : 'pill')}
          />
        </ChartSize>
        <ChartSize scale={0.4}>
          <XYChart
            xLabel="α"
            xRange={[0, top * 1.2]}
            yLabel="total leaf impurity R"
            series={[
              {
                name: 'R(T_α)',
                type: 'line',
                x: alphas.slice(0, -1).flatMap((a, i) => [a, alphas[i + 1]]),
                y: impurities.slice(0, -1).flatMap((r) => [r, r]),
              },
              { name: 'leaves', type: 'scatter', x: alphas, y: impurities, muted: true },
            ]}
            handles={[{ kind: 'x', at: alpha, onDrag: (a) => setAlpha(Math.max(0, a)), label: 'α' }]}
            formatY={(v) => `${formatValue(v)} (${leaves[alphas.findLastIndex((a) => a <= alpha)] ?? ''} leaves)`}
          />
        </ChartSize>
      </div>
    </Figure>
  )
}

// ── Boosting round by round ──────────────────────────────────────────────────────────────────────────────────────

export function BoostingSpecimen() {
  const [method, setMethod] = useState<'adaboost' | 'gradient'>('adaboost')
  const [dataset, setDataset] = useState<DatasetName>('circles')
  const [round, setRound] = useState(10)
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
  const cols = columns(data.x)
  const labels = toFlat(data.y!)
  const weights = state ? toFlat(state.sampleWeights) : null
  const heavy = weights
    ? weights
        .map((w, i) => [w, i])
        .sort((a, b) => b[0] - a[0])
        .slice(0, 15)
        .map((p) => p[1])
    : []
  const overlay: HeatmapOverlay[] = [
    { name: 'rows', type: 'scatter', x: cols.x0, y: cols.x1, group: labels, groupNames: data.meta.labelNames },
    ...(heavy.length
      ? [
          {
            name: 'heaviest weights',
            type: 'scatter' as const,
            x: heavy.map((i) => cols.x0[i]),
            y: heavy.map((i) => cols.x1[i]),
            emphasis: true,
          },
        ]
      : []),
  ]
  const loss = method === 'adaboost' ? toFlat(ada.training.series.trainingError) : toFlat(gb.training.series.loss)
  return (
    <Figure
      title="Boosting, round by round"
      description="Each round adds one stump fitted to what the ensemble so far gets wrong: reweighted rows for AdaBoost, pseudo-residuals for gradient boosting."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · setup">
            <Select
              label="method"
              value={method}
              onChange={setMethod}
              options={[
                { value: 'adaboost', label: 'AdaBoost (SAMME)' },
                { value: 'gradient', label: 'gradient boosting (logistic)' },
              ]}
            />
            <Select
              label="dataset"
              value={dataset}
              onChange={setDataset}
              options={DATASET_OPTIONS.filter((o) => o.value !== 'blobs')}
            />
          </ControlRow>
          <ControlRow label="2 · rounds">
            <Player value={r} onChange={setRound} count={rounds + 1} label="round" />
          </ControlRow>
        </>
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
      caption="The colour is the ensemble's score after the chosen round (vote margin for AdaBoost, log-odds for gradient boosting), with its zero contour as the boundary. For AdaBoost, the ringed rows carry the most weight going into the next round: they sit where the current boundary is wrong. One stump draws one axis-aligned cut; eighty of them trace the circle."
    >
      <Heatmap
        x={toFlat(g.x)}
        y={toFlat(g.y)}
        z={z}
        scale="diverging"
        contours={{ levels: [0] }}
        overlay={overlay}
        equalAspect
        xLabel="x₀"
        yLabel="x₁"
        valueLabel="score"
        range={method === 'adaboost' ? [-1, 1] : [-6, 6]}
      />
    </Figure>
  )
}
